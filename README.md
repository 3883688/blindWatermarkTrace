# BlindWatermarkTrace V4

BlindWatermarkTrace 是一套基于 FastAPI、PostgreSQL/pgvector 和 Vue 3 的图片盲水印生成、检测与证据追踪系统。当前生产分支为 `V4-MASTER`，生产链路只生成和认证 V4 水印。

系统为每张图片生成可认证的追溯记录，并保存原图、水印成品、文件指纹、所有者和证据编号。检测时优先使用精确文件指纹，再进入 V4 盲水印、几何配准和视觉召回流程。

![BlindWatermarkTrace 界面](demo/1.jpg)

## 当前能力

- V4 认证盲水印：DCT 码字与 FFT 同步导频组合，支持压缩、缩放、旋转和局部裁剪场景。
- 文件指纹快速命中：使用 MD5 与 SHA-256，命中时无需执行完整图像推理。
- 视觉召回与配准：DINOv2 用于候选召回，SuperPoint + LightGlue 用于局部特征匹配。
- 人物关键区域追溯增强：可选识别人脸及特定区域，对相关 V4 tile 增强嵌入。
- 自适应计算后端：`auto` 模式检测 NVIDIA CUDA；不可用或烟测失败时回退 CPU。
- JPEG 成品输出：默认质量 80，使用 4:4:4 色度采样，前端可设置 60 至 95。
- 可见版权水印：默认关闭，可配置文字、透明度、铺排密度和角落版权块。
- 用户隔离：普通用户只能查看和检测自己权限范围内的数据，管理员可以跨用户查看。
- 管理与统计：图片记录、证据、用户、角色权限、今日水印数和今日检测次数。
- 受控媒体地址：原图和成品使用签名、限时、不可枚举的媒体访问 URL。

## V4 工作流程

```text
生成：登录 -> 上传原图 -> 创建 source group / trace ID
     -> 可选人物关键区域识别与明水印
     -> FFT 同步导频 + DCT 认证码字
     -> JPEG 编码 -> 保存证据和媒体 -> 返回限时 URL

检测：登录 -> 上传图片或提交 URL -> 文件双哈希匹配
     -> V4 认证检测 -> 几何校正 / 视觉候选
     -> 所有者权限校验 -> 返回证据记录
```

当前编码器标识：

```text
hmac64_rs_16_8_split_repeat_sync_v4
```

V4 使用服务端 `WATERMARK_AUTH_KEY` 生成认证标签，再通过 Reed-Solomon 冗余、分块 DCT 和全图重复编码写入亮度通道。FFT 导频用于估计旋转、缩放和平移，检测器校正网格后再认证码字。密钥丢失或被更换后，历史水印的认证能力会受影响。

## 技术栈

- Python 3.10+、FastAPI、Uvicorn、SQLAlchemy
- PostgreSQL 16、pgvector、psycopg 3
- Pillow、OpenCV、NumPy、Reed-Solomon
- ONNX Runtime、DINOv2、SuperPoint/LightGlue、YOLOX、DWPose
- 可选 CuPy + CUDA 12
- Vue 3、Element Plus、Vite
- pytest、Vitest、Playwright

## 项目结构

```text
main.py                         FastAPI 入口（main:app）
index.html                      生产前端入口
assets/                         已构建的前端资源
frontend/                       Vue 3 前端源码和测试
models/                         V4 与人物区域识别 ONNX 模型
trace_app/
  api/                          登录、V4、兼容 API、媒体和用户路由
  auth/                         会话、用户与角色权限
  database/                     数据库连接、基础表和仓储
  imaging/                      图片 IO、指纹、明水印和视觉匹配
  management/                   图片管理与统计
  v4/                           V4 领域、生产服务、模型和 PostgreSQL 仓储
watermark_v4/                   DCT、FFT、码字和 CPU/CUDA 计算后端
tools/                          部署、数据库迁移、模型和 GPU 工具
tests/                          后端、API、模型与鲁棒性测试
release/                        发布包（不得包含 .env 和业务数据）
```

## 必需模型

生产环境的 `models/` 至少包含以下文件及对应 SHA256 文件：

| 文件 | 用途 |
| --- | --- |
| `dinov2-small.onnx` | 全局视觉特征和候选召回 |
| `superpoint_lightglue_pipeline.onnx` | 局部特征匹配与几何配准 |
| `yolox-tiny-humanart-person.onnx` | 人物检测 |
| `dwpose-s-wholebody.onnx` | 人脸与特定区域关键点估计 |

人物区域增强关闭时，后两个模型不会进入生成路径。模型缺失或推理异常时，该增强链路会回退到 OpenCV 检测器，主水印生成仍继续执行。

ONNX 文件本身已压缩，完整发布包通常约 177 MB；不携带模型的代码更新包约 6 至 8 MB。只有在服务器已具备且校验通过的同版本模型时，才应使用精简更新包。

## 本地开发

### 后端

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Linux/macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

填写 `.env` 后启动：

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

- Web：<http://127.0.0.1:8000>
- OpenAPI：<http://127.0.0.1:8000/docs>
- OpenAPI JSON：<http://127.0.0.1:8000/openapi.json>

完整 V4 生产模式要求 PostgreSQL 和 pgvector。SQLite 只适合部分开发或自动化测试，不是生产数据库。

### 前端

```bash
npm --prefix frontend install
npm --prefix frontend run dev
npm --prefix frontend run build
```

生产服务直接提供根目录 `index.html` 和 `assets/`。前端修改后必须执行构建，并确认生成的 `assets/app/app.js` 已更新。

## 环境配置

生产环境最低配置示例：

```dotenv
APP_NAME=WatermarkSystem
ENVIRONMENT=production
ADMIN_USER=admin
ADMIN_PASS=replace-with-a-strong-password
DB_ENABLED=true
DB_URL=postgresql+psycopg://trace:password@postgres:5432/trace
UPLOAD_DIR=./uploads
DATA_DIR=./data
V4_MODEL_MANIFEST_PATH=./models/v4-manifest.json
ROBUST_WATERMARK_VERSION=4
ROBUST_WATERMARK_STRENGTH=0.74
WATERMARK_AUTH_KEY=replace-with-at-least-32-random-bytes
TRACE_COMPUTE_DEVICE=auto
VISIBLE_COPYRIGHT_ENABLED=false
```

生成密钥：

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

主要变量：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `ENVIRONMENT` | `production` 时强制 PostgreSQL + pgvector | `development` |
| `DB_URL` | SQLAlchemy PostgreSQL 连接串 | 空 |
| `ADMIN_USER` / `ADMIN_PASS` | 初始管理员账号 | 空 |
| `UPLOAD_DIR` | 原图、成品和缩略图目录 | `./uploads` |
| `DATA_DIR` | 本地运行数据目录 | `./data` |
| `WATERMARK_AUTH_KEY` | V4 认证密钥，至少 32 字节 | 空 |
| `TRACE_COMPUTE_DEVICE` | `auto`、`cpu` 或 `cuda` | `auto` |
| `MEDIA_PUBLIC_BASE_URL` | 媒体签名 URL 的公开地址前缀 | 空 |
| `V4_SYNC_TIMEOUT_SECONDS` | 同步任务硬超时 | `300` |
| `V4_DEEP_TIMEOUT_SECONDS` | 深度任务硬超时 | `1000` |
| `VISIBLE_COPYRIGHT_ENABLED` | 默认是否启用明水印 | `false` |
| `VISIBLE_COPYRIGHT_TEXT` | 默认明水印文字 | `© QQ:757675150` |
| `VISIBLE_COPYRIGHT_OPACITY` | 明水印透明度，范围 0.02 至 0.90 | `0.16` |
| `WATERMARK_DETECTION_BUDGET_SECONDS` | 兼容检测时间预算 | `5` |

应用启动时会自动执行数据库初始化：创建 pgvector 扩展、缺失表和 V4 索引，并校验 HNSW cosine 索引。生产数据库账号必须具有相应权限。数据库初始化不会清空已有记录。

## 前端生成参数

| 参数 | 表单字段 | 默认值 | 范围或行为 |
| --- | --- | --- | --- |
| 明水印 | `copyright_enabled` | 关闭 | 每次生成可独立设置 |
| JPEG 输出质量 | `output_quality` | `80` | 60 至 95 |
| 同步导频强度 | `pilot_amplitude` | `0.75` | 0.25 至 1.25 |
| 人物关键区域追溯增强 | `protected_region_enhancement` | 关闭 | 识别人脸及特定区域后增强相关 tile |

明水印与人物关键区域增强均默认关闭。JPEG 成品使用用户选择的质量值；非法值回退为 80，超出范围时限制到 60 至 95。

## GPU 加速

基础依赖只安装 CPU 运行时：

```bash
python -m pip install -r requirements.txt
```

在具备 NVIDIA 驱动和 CUDA 12 的主机上执行自动检测安装：

```bash
python tools/install_optional_gpu.py \
  --python .venv/bin/python \
  --requirements requirements-gpu.txt
```

Windows 将 Python 参数改为 `.venv\Scripts\python.exe`。

- `TRACE_COMPUTE_DEVICE=auto`：检测设备并验证计算结果，适合生产默认值。
- `TRACE_COMPUTE_DEVICE=cpu`：强制 CPU，不导入 CuPy。
- `TRACE_COMPUTE_DEVICE=cuda`：请求 CUDA；设备或运行时不可用时仍回退 CPU。

CuPy 用于 V4 数值计算加速。DINOv2 和 LightGlue 当前适配器固定使用 ONNX Runtime CPU Provider；人物区域模型会在 ONNX Runtime 暴露 CUDA Provider 时优先使用 CUDA，否则使用 CPU。

Docker 部署还需要 NVIDIA Container Toolkit，并把 GPU 暴露给应用容器；仅在宿主机安装 CUDA 依赖不会自动把 GPU 注入容器。

## API

先登录并保存 Cookie：

```bash
curl -c cookies.txt -X POST http://127.0.0.1:8000/auth/login \
  -F "username=admin" \
  -F "password=replace-with-a-strong-password"
```

使用兼容前端契约生成 V4 水印：

```bash
curl -b cookies.txt -X POST http://127.0.0.1:8000/api/watermark/embed \
  -F "file=@./demo/1.jpg" \
  -F "output_quality=80" \
  -F "pilot_amplitude=0.75" \
  -F "copyright_enabled=false" \
  -F "protected_region_enhancement=false"
```

检测图片：

```bash
curl -b cookies.txt -X POST http://127.0.0.1:8000/api/watermark/extract \
  -F "file=@./watermarked.jpg"
```

主要接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/auth/login`、`/auth/logout` | 登录和退出 |
| POST | `/api/watermark/embed` | 前端兼容的 V4 生成接口 |
| POST | `/api/watermark/extract` | 上传文件检测 |
| POST | `/api/watermark/extract-url` | 下载 URL 并检测 |
| GET | `/api/images` | 当前权限范围内的图片和统计 |
| GET | `/api/dashboard-stats` | 今日水印数和今日检测次数 |
| POST | `/api/v4/generate` | 原生 V4 生成接口 |
| POST | `/api/v4/detect`、`/api/v4/detect-url` | 原生 V4 检测接口 |
| GET | `/api/v4/records` | V4 记录列表 |
| GET | `/api/v4/capabilities` | 模型能力状态 |
| POST/GET/DELETE | `/api/v4/jobs` | 深度检测任务 |
| GET | `/api/media/{media_id}` | 签名媒体访问 |
| GET/POST/PUT/DELETE | `/api/users`、`/api/roles` | 用户和角色管理 |

管理员查询和统计汇总所有用户；普通用户只看到自己的记录。跨用户能力由服务端角色控制，不能通过普通用户提交参数绕过。

## 测试与验收

后端：

```bash
python -m pytest -q
```

前端单元测试和构建：

```bash
npm --prefix frontend test -- --run
npm --prefix frontend run build
```

浏览器测试：

```bash
npm --prefix frontend run test:ui
```

部分模型、商业样本和鲁棒性基准测试需要本地 ONNX 文件或私有图片数据集。

## Docker 更新部署

当前生产服务器目录：

```text
/opt/trace-v4-docker-20260726-164548
```

使用 `release/` 下的 V4-MASTER Docker 更新包：

```bash
unzip trace-v4-master-update-YYYYMMDD-HHMMSS.zip
cd trace-v4-master-update-YYYYMMDD-HHMMSS
chmod +x update.sh
sudo TRACE_TARGET_DIR=/opt/trace-v4-docker-20260726-164548 \
  TRACE_APP_SERVICE=app \
  ./update.sh
```

更新脚本应完成 payload 哈希校验、旧文件备份、容器停止、代码与模型覆盖、镜像重建、数据库初始化、容器健康检查和失败回滚。它必须保留目标目录中的 `.env`、数据库卷、上传目录和 Compose 配置。

更新后检查：

```bash
cd /opt/trace-v4-docker-20260726-164548
docker compose ps app
docker compose logs --tail=100 app
```

如果访问日志仍显示旧的 `/opt/trace-v4-centos-*` 路径，说明反向代理仍指向旧 systemd 服务，需要核对 Nginx upstream 与 Docker 映射端口。

## 安全边界

不得提交或放入发布包：

- `.env`、管理员密码、数据库连接串和 `WATERMARK_AUTH_KEY`
- PostgreSQL 数据目录、`data/`、`uploads/` 和真实业务图片
- 登录 Cookie、媒体签名 URL、生产日志和私有测试数据
- `.venv/`、`node_modules/`、缓存和临时输出

生产环境应使用 HTTPS、强管理员密码、随机认证密钥和最小权限数据库账号。媒体签名 URL 是临时凭证，不应写入公开日志或长期保存。

## 已知限制

- 盲水印不能保证抵抗所有重绘、严重模糊、大面积遮挡或极小裁剪。
- 检测结果应结合置信度、证据链和人工审核，不能单独视为法律结论。
- V4 格式、认证密钥和模型版本属于生产契约，变更前必须规划兼容与迁移。
- GPU 自动回退保证可运行，不保证每种图片尺寸下 GPU 都一定比 CPU 更快。

## License

当前仓库未声明开源许可证。对外发布前应根据实际授权方式补充 `LICENSE`。
