# Trace System V4 CentOS 一键部署

部署方式沿用原 V1 的 CentOS + systemd + 现有 MySQL 流程，实际运行水印算法固定为 V4。

## 前提

- CentOS/RHEL，具备 `dnf` 或 `yum`。
- Python 3.10 或更高版本；脚本会自动选择 `python3.10` 至 `python3.13`，也可通过 `PYTHON_BIN` 指定。
- 服务器已有 MySQL/MariaDB，并且 `.env` 中的 `DB_URL` 可以连接。
- 脚本不会安装 MySQL、建库或创建数据库用户；应用和迁移命令会创建业务表。
- 默认服务名 `trace-system`，默认端口 `6868`。

## 解压与一键部署

发布 ZIP 不包含 `.env`、`data/`、`uploads/`、数据库文件或历史备份。覆盖解压代码不会覆盖这些业务文件。
应用入口仍为 `main:app`；`main.py` 仅保留兼容导入，完整应用实现位于 `trace_app/` 模块树中，发布包会递归包含该模块树。

```bash
mkdir -p /opt/trace
unzip -o trace-v4-centos-20260717.zip -d /opt/trace
cd /opt/trace
chmod +x deploy.sh
cp .env.example .env
chmod 600 .env
# 编辑 .env，填写 ADMIN_USER、ADMIN_PASS 和 DB_URL
./deploy.sh migrate-data
sudo ./deploy.sh install-service
```

`migrate-data` 读取服务器已有 `data/` 下的 5 个 JSON，在一个事务中写入
`image_records`、`users`、`roles` 和 `stats`。用户密码在入库前转换为带随机盐的
scrypt 哈希。数据库反查核验成功后，源 JSON 会备份到项目目录外的
`trace-private-migration-backups/`，随后才从 `data/` 删除。

发布包不携带业务 JSON。若目标服务器没有旧部署目录，需要通过 Git 之外的安全渠道
将 5 个 JSON 放入 `/opt/trace/data/`，执行迁移后立即清理传输副本。

脚本依次执行：

1. 安装 Python、pip、firewalld 和 tar 等运行依赖，不安装数据库服务。
2. 将现有 `.env`、`data/`、`uploads/` 备份到 `backups/deploy/`。
3. 保留已有 `.env`；不存在时从 `.env.example` 创建。
4. 强制写入唯一的 `ROBUST_WATERMARK_VERSION=4`。
5. 已有有效 `WATERMARK_AUTH_KEY` 保持不变；缺失、过短或重复时生成 48 个安全随机字节的 Base64 密钥，密钥不会输出到终端或日志。
6. 使用现有 MySQL 应用账号执行 `SELECT 1`，连接失败则在重启前停止部署。
7. 创建数据库业务表、`.venv`，安装依赖、注册并重启 systemd 服务。
8. 轮询 `http://127.0.0.1:6868/`，必须返回 HTTP 200 才判定部署成功。

重复执行部署命令时，数据库记录、`.env`、`WATERMARK_AUTH_KEY` 和 `uploads/` 均保持不变。

## 只检查数据库

```bash
cd /opt/trace
./deploy.sh check-db
```

该命令只验证现有 MySQL 连接，不会建库或修改数据。

## 服务操作

```bash
systemctl start trace-system
systemctl stop trace-system
systemctl restart trace-system
systemctl status trace-system
journalctl -u trace-system -f
```

访问地址：

```text
http://<服务器IP>:6868
```

## 部署失败

脚本会输出 `systemctl status`、最近 80 行 journal 日志和本次备份路径。修复配置后重复执行：

```bash
sudo ./deploy.sh install-service
```

部署前备份位于：

```text
/opt/trace/backups/deploy/pre-deploy-<UTC时间>.tar.gz
```

## 自定义端口与服务名

```bash
sudo PORT=6868 SERVICE_NAME=trace-system ./deploy.sh install-service
```

## 必填私有配置

以下值只允许保存在权限受限且被 Git 忽略的 `.env` 中：

- `DB_URL`
- `ADMIN_USER`
- `ADMIN_PASS`

`.env.example` 只提供空字段，不提供账号、密码或连接串默认值。
