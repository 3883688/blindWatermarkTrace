# Trace System V4 CentOS 一键部署

部署方式沿用原 V1 的 CentOS + systemd + 现有 MySQL 流程，实际运行水印算法固定为 V4。

## 前提

- CentOS/RHEL，具备 `dnf` 或 `yum`。
- Python 3.10 或更高版本；脚本会自动选择 `python3.10` 至 `python3.13`，也可通过 `PYTHON_BIN` 指定。
- 服务器已有 MySQL/MariaDB，并且 `.env` 中的 `DB_URL` 可以连接。
- 脚本不会安装、清空或修改数据库结构，不会建库、建用户。
- 默认服务名 `trace-system`，默认端口 `6868`。

## 解压与一键部署

发布 ZIP 不包含 `.env`、`data/`、`uploads/`、数据库文件或历史备份。覆盖解压代码不会覆盖这些业务文件。

```bash
mkdir -p /opt/trace
unzip -o trace-v4-centos-20260715.zip -d /opt/trace
cd /opt/trace
chmod +x deploy.sh
sudo ./deploy.sh install-service
```

脚本依次执行：

1. 安装 Python、pip、firewalld 和 tar 等运行依赖，不安装数据库服务。
2. 将现有 `.env`、`data/`、`uploads/` 备份到 `backups/deploy/`。
3. 保留已有 `.env`；不存在时从 `.env.example` 创建。
4. 强制写入唯一的 `ROBUST_WATERMARK_VERSION=4`。
5. 已有有效 `WATERMARK_AUTH_KEY` 保持不变；缺失、过短或重复时生成 48 个安全随机字节的 Base64 密钥，密钥不会输出到终端或日志。
6. 使用现有 MySQL 应用账号执行 `SELECT 1`，连接失败则在重启前停止部署。
7. 创建 `.venv`、安装依赖、注册并重启 systemd 服务。
8. 轮询 `http://127.0.0.1:6868/`，必须返回 HTTP 200 才判定部署成功。

重复执行部署命令时，数据库记录、`.env`、`WATERMARK_AUTH_KEY`、`data/` 和 `uploads/` 均保持不变。

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

## 保留的默认配置

- Database host: `127.0.0.1`
- Database: `trace`
- Database username: `REMOVED`
- Database password: `REMOVED_PASSWORD`
- Admin username: `REMOVED_ADMIN_USER`
- Admin password: `REMOVED_PASSWORD`
