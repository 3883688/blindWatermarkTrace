# Trace System

基于 FastAPI 的图片水印生成与追溯系统。系统为图片生成多层隐式水印、保存证据记录，并支持对原图、水印图及经过处理的图片进行归因追溯。

![Trace System 演示图](demo/1.jpg)

## 核心能力

- 生成可见版权、水印载荷与多种隐式水印层。
- 支持 V4 认证水印、DCT/DWT/FFT、LSB、小裁剪追溯与点阵追溯。
- 对上传文件先进行精确文件指纹比对，再进入图片解码与水印检测流程。
- 提供图片管理、检测统计、用户管理和角色权限接口。
- 使用证据 UUID、追溯 ID、原图和水印图 URL 保存归因证据。

## 精确文件追溯

上传图片的追溯顺序如下：

1. 读取上传文件字节，并计算 MD5 和 SHA-256。
2. 用 MD5 从已登记的原图和水印图中筛选候选记录。
3. 仅当候选记录的 SHA-256 同时一致时，立即返回成功结果。
4. 原图和水印图的精确匹配都属于成功归因，不会解码图片或执行水印算法。
5. 不含 MD5 的历史记录继续使用 SHA-256 文件指纹匹配。
6. 未命中精确文件指纹时，才进入像素指纹和水印检测流水线。

这个流程在响应中以 `matched_hash_type=file_md5_sha256` 标识新记录的双哈希命中，以 `file_sha256` 标识历史记录命中。

## 架构

```text
main.py                         兼容入口
└── trace_app/
    ├── application.py          FastAPI 装配、静态资源与路由注册
    ├── api/                    HTTP 路由层
    ├── dependencies.py         FastAPI 依赖注入
    ├── auth/                   登录、用户与角色服务
    ├── database/               数据库连接与 Repository
    ├── watermark/              水印生成、检测和算法编排
    ├── imaging/                图片 IO、指纹、特征匹配和可见水印
    ├── management/             图片管理和仪表盘统计
    ├── config.py               环境变量与运行参数
    └── compat.py               旧版 main API 兼容层
```

典型请求链：

```text
HTTP 请求 → api 路由 → Depends 服务注入 → Service → Repository / 算法模块
```

## 本地启动

### 前置条件

- Python 3.10 或更高版本。
- 可选：MySQL/MariaDB。开发环境可以使用 SQLite。

### 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，本地 SQLite 示例：

```dotenv
ADMIN_USER=admin
ADMIN_PASS=change-this-password
DB_URL=sqlite+pysqlite:///data/trace-dev.db
ROBUST_WATERMARK_VERSION=4
WATERMARK_AUTH_KEY=replace-with-a-random-secret-of-at-least-32-bytes
```

生成本地密钥：

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

启动应用：

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。

## API 概览

| 分组 | 接口 |
| --- | --- |
| 认证 | `POST /auth/login` |
| 水印 | `POST /api/watermark/embed`、`POST /api/watermark/extract`、`POST /api/watermark/extract-url` |
| 图片 | `GET /api/images`、`DELETE /api/images/{image_id}` |
| 用户与角色 | `GET /api/roles`、`PUT /api/roles/{role_key}`、`GET/POST /api/users`、`PUT/DELETE /api/users/{username}` |
| 仪表盘 | `GET /api/dashboard-stats` |

接口参数以各路由模块为准：`trace_app/api/`。

## 测试

运行全部测试：

```powershell
python -m pytest -q
```

运行与文件指纹追溯直接相关的测试：

```powershell
python -m pytest tests/test_application_structure.py::test_imaging_fingerprints_hashes_file_bytes_with_md5 tests/test_watermark_v4_api.py::test_v4_exact_file_fingerprints_succeed_without_image_decode -q
```

## 生产部署

CentOS/RHEL、systemd、MySQL/MariaDB 以及迁移流程请参阅 [README_DEPLOY.md](README_DEPLOY.md)。该文档是生产部署的唯一详细说明。

## 安全与数据边界

- 把 `ADMIN_USER`、`ADMIN_PASS`、`DB_URL` 和 `WATERMARK_AUTH_KEY` 放在 Git 忽略的 `.env` 中。
- 不要提交 `data/`、`uploads/`、数据库文件、特征索引或真实业务图片。
- 生产环境使用独立数据库账号，并限制其数据库权限范围。
