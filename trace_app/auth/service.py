"""认证与用户/角色管理的服务层。

承载 ``/auth/login``、``/api/roles``、``/api/users`` 背后的全部业务规则，
让路由层保持薄委托。两点设计需要留意：

**会话存放在进程内存。** ``sessions`` 是一个 ``token -> user_id`` 的字典，
不落库也不跨进程共享——这意味着服务重启即全员掉线，多副本部署必须配置粘性会话
或改用外部会话存储。构造时可注入自定义字典，测试因此能预置登录态。

**仓储可以缺席。** ``repository`` 允许为 ``None``（数据库尚未就绪时），
只有真正访问数据时才通过 :meth:`_require_repository` 抛 503，
从而避免应用启动阶段因数据库未连上而整体失败。
"""

import uuid
from typing import Any

from fastapi import HTTPException

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.config import MENU_LABELS
from trace_app.database.repositories import Repository


class AuthService:
    """认证服务：登录、令牌解析、角色与用户的增删改查。"""

    def __init__(
        self,
        repository: Repository | None = None,
        *,
        sessions: dict[str, int] | None = None,
    ) -> None:
        """
        :param repository: 数据仓储；``None`` 表示数据库暂不可用（延迟到访问时才报错）。
        :param sessions: 令牌表，外部注入时**共享同一个字典对象**，
            这样应用重建服务实例后已登录用户不会掉线。
        """
        self.repository = repository
        self.sessions = {} if sessions is None else sessions

    def _require_repository(self) -> Repository:
        """取仓储；未配置时以 503 结束请求，而不是抛 ``AttributeError``。"""
        if self.repository is None:
            raise HTTPException(status_code=503, detail="数据库不可用")
        return self.repository

    def public_users(self, users: dict[str, Any]) -> dict[str, dict[str, str]]:
        """把用户表投影成"只含角色"的对外形状。

        这是一层显式的**减法**：仓储行里还有密码哈希、盐、创建时间等字段，
        经过本方法后只剩 ``role``，从源头杜绝凭据随列表接口外泄。
        角色缺失时统一落到 ``operator``（最小权限）。
        """
        return {
            username: {"role": str(info.get("role") or "operator")}
            for username, info in users.items()
        }

    def allowed_menu_keys(self, menus: Any) -> list[str]:
        """过滤出系统真实存在的菜单键。

        非列表入参（``None``、字符串、字典）一律当作空授权处理，避免把前端
        传来的脏数据写进权限表。保留原始顺序，因为菜单顺序即前端展示顺序。
        """
        if not isinstance(menus, list):
            return []
        return [key for key in menus if key in MENU_LABELS]

    def role_for_username(self, username: str) -> str:
        """查某个用户名的角色，未知用户按最小权限 ``operator`` 处理。"""
        users = self._require_repository().read_users()["users"]
        return str(users.get(username, {}).get("role") or "operator")

    def login(self, username: str, password: str) -> dict[str, Any]:
        """校验凭据并签发会话令牌。

        :return: ``token``/``username``/``role``/``menus`` 四件套，
            前端拿 ``menus`` 直接渲染侧边栏，无需再发一次权限请求。

        用户名不存在与密码错误共用同一句 401 文案，不区分二者，
        防止通过报错差异枚举有效用户名。
        """
        repository = self._require_repository()
        identity = repository.authenticate_user(username, password)
        if identity is None:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        role = str(identity["role"])
        roles = repository.read_roles()["roles"]
        # 菜单来自角色配置，且再过一道白名单：角色表里的历史脏键不会下发给前端。
        menus = self.allowed_menu_keys(roles.get(role, {}).get("menus", []))
        # 令牌只是随机串，不携带任何身份信息——权限一律回表查，
        # 因此改角色/删用户能立即生效，不存在"旧令牌仍带旧权限"的窗口。
        token = f"local-{uuid.uuid4().hex}"
        self.sessions[token] = int(identity["id"])
        return {
            "token": token,
            "username": str(identity["username"]),
            "role": role,
            "menus": menus,
        }

    def resolve_token(self, token: str) -> AuthenticatedUser:
        """把令牌还原成当前用户身份，供依赖注入层做鉴权。

        每次都回库读用户，而不是缓存登录时的快照——用户被删或改角色后，
        下一次请求就会失效或按新角色执行。

        令牌存在但用户已不存在时，顺手把这条会话从表中清掉再抛 401，
        避免内存里堆积永远解析不出身份的僵尸令牌。
        """
        user_id = self.sessions.get(token)
        identity = (
            None
            if user_id is None
            else self._require_repository().get_user_by_id(user_id)
        )
        if identity is None:
            self.sessions.pop(token, None)
            raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
        return AuthenticatedUser(
            id=int(identity["id"]),
            username=str(identity["username"]),
            role=str(identity["role"]),
        )

    def list_roles(self) -> dict[str, Any]:
        """返回全部角色及其菜单授权，并附上菜单键到中文名的字典。

        ``menus`` 字段是**全量可选项**（``MENU_LABELS``），前端据此渲染
        权限勾选框；``roles`` 里才是各角色的实际勾选状态。
        """
        roles = self._require_repository().read_roles()["roles"]
        return {"menus": MENU_LABELS, "roles": roles}

    def update_role(self, role_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        """覆盖式更新某角色的菜单授权，并回传更新后的完整角色表。

        写入前先过 :meth:`allowed_menu_keys`：未知键被丢弃而非报错，
        这样前后端菜单版本不一致时是"少授权"而不是"整个请求失败"。
        """
        repository = self._require_repository()
        roles = repository.read_roles()["roles"]
        if role_key not in roles:
            raise HTTPException(status_code=404, detail="角色不存在")
        repository.update_role_menus(
            role_key, self.allowed_menu_keys(payload.get("menus"))
        )
        # 回读一次而不是就地拼装，保证返回的就是库里的最终状态。
        return self.list_roles()

    def list_users(self) -> dict[str, Any]:
        """返回脱敏后的用户表与角色表（前端用户管理页一次性取全）。"""
        repository = self._require_repository()
        return {
            "users": self.public_users(repository.read_users()["users"]),
            "roles": repository.read_roles()["roles"],
        }

    def create_user(self, payload: dict[str, Any]) -> dict[str, Any]:
        """新建用户。

        校验顺序刻意是"先必填、再合法性、最后唯一性"，对应 400/400/400/409
        四种回执；重名单独用 409 是为了让前端能区分"填错了"和"已存在"。
        角色缺省为 ``operator``，即新账号默认最小权限。
        """
        repository = self._require_repository()
        # strip 用户名但**不** strip 密码：密码里的空格是有效字符。
        username = str(payload.get("username") or "").strip()
        password = str(payload.get("password") or "")
        role = str(payload.get("role") or "operator")
        roles = repository.read_roles()["roles"]
        if not username:
            raise HTTPException(status_code=400, detail="请输入用户名")
        if not password:
            raise HTTPException(status_code=400, detail="请输入密码")
        if role not in roles:
            raise HTTPException(status_code=400, detail="角色不存在")
        if username in repository.list_users():
            raise HTTPException(status_code=409, detail="用户已存在")
        repository.create_user(username, password, role)
        return {"users": repository.list_users(), "roles": roles}

    def update_user(self, username: str, payload: dict[str, Any]) -> dict[str, Any]:
        """更新用户——当前**仅支持改角色**。

        ``payload`` 里的其他键会被忽略；改密码走独立流程，不从这里进，
        以免用户管理接口意外具备重置任意账号密码的能力。
        """
        repository = self._require_repository()
        users = repository.list_users()
        if username not in users:
            raise HTTPException(status_code=404, detail="用户不存在")
        role = str(payload.get("role") or "")
        roles = repository.read_roles()["roles"]
        if role not in roles:
            raise HTTPException(status_code=400, detail="角色不存在")
        repository.update_user_role(username, role)
        return {"users": repository.list_users(), "roles": roles}

    def delete_user(self, username: str) -> dict[str, Any]:
        """删除用户，并回传更新后的用户表与角色表。

        仓储的 ``delete_user`` 返回布尔值表示是否真的删掉了一行，
        据此把"用户不存在"翻成 404，而不是静默返回成功。

        注意：该用户已签发的令牌不会在此清理，但因为
        :meth:`resolve_token` 每次都回库校验，删除后下一次请求即失效。
        """
        repository = self._require_repository()
        if not repository.delete_user(username):
            raise HTTPException(status_code=404, detail="用户不存在")
        return {
            "users": repository.list_users(),
            "roles": repository.read_roles()["roles"],
        }
