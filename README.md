# BlindWatermarkTrace

基于 FastAPI 的图片盲水印生成、检测与来源追踪系统。

项目将业务溯源 ID 编码为经过认证的盲水印，嵌入图片后保存原图、水印图、文件指纹和证据记录。当图片被重新上传、压缩、缩放、旋转或发生局部裁剪时，系统可以通过文件指纹、水印检测和视觉匹配等路径判断图片来源。

> 当前默认鲁棒水印版本为 V4，主链路是 DCT 认证码字 + FFT 同步导频。

![BlindWatermarkTrace 演示图](demo/1.jpg)

## 功能特性

- 生成和提取 V4 认证水印、DCT/DWT/FFT 频域水印、LSB 水印、点阵追踪和小裁剪追踪。
- 可选生成可见版权水印，用于人工识别和版权提示。
- 先进行 MD5 + SHA-256 文件指纹快速匹配，再进入图像解码和水印检测流程。
- 支持上传文件和图片 URL 两种追踪方式。
- 保存溯源 ID、证据 UUID、原图、水印图、检测结果和置信度。
- 提供图片管理、检测统计、用户管理、角色权限和受控媒体下载接口。
- 支持 SQLite、MySQL/MariaDB、PostgreSQL，以及可选的 NVIDIA CUDA/CuPy 加速。

## 工作流程

```text
生成：上传原图 -> 生成 trace_id -> 嵌入水印 -> 保存证据 -> 返回限时下载地址

追踪：上传图片 -> 文件指纹匹配 -> V4 检测 -> 其他水印检测 -> 视觉匹配 -> 返回归因证据
```

精确文件匹配命中时，不需要解码图片或运行水印算法。响应中的
`matched_hash_type=file_md5_sha256` 表示 MD5 + SHA-256 双哈希命中，`file_sha256` 表示历史记录的 SHA-256 命中。

## 使用的算法

### V4 认证盲水印

V4 的编码器标识为：

```text
hmac32_rs_8_4_full_repeat_sync_v4
```

数据编码流程：

```text
trace_id -> HMAC-SHA256 截取 4 字节认证标签
         -> Reed-Solomon RS(8,4) 得到 8 字节码字
         -> 64 个比特 -> 按 tile 位置进行相位置换
         -> 写入图片亮度通道
```

核心设计：

- **HMAC-SHA256**：由服务端密钥和 `trace_id` 生成认证标签，无法只靠公开算法伪造有效水印。
- **Reed-Solomon RS(8,4)**：4 字节数据加 4 字节校验；检测时结合置信度尝试擦除，提升压缩和局部损坏后的恢复能力。
- **分块 DCT**：图片按 128x128 tile 处理，每个 tile 划分为 8x8 个 16x16 cell，每个 cell 承载 1 bit。
- **中频系数对比较调制**：通过两组中频 DCT 系数的相对大小表示 0/1，实现无需原图的盲提取，并平衡画质和压缩鲁棒性。
- **亮度通道嵌入**：只修改 Y 通道，减少色度二次采样的影响。
- **全图重复编码**：多个完整 tile 重复承载同一码字，保留足够 tile 时可支持局部裁剪后的恢复。
- **相位置换**：不同 tile 使用不同的确定性 bit 置换，降低固定模式带来的误检风险。

### FFT 同步导频

缩放和旋转会破坏 DCT 网格对齐。V4 在亮度通道叠加四个已知频率的低强度导频，检测时分析 FFT 频谱峰值：

1. 粗略搜索旋转角和缩放比例。
2. 在候选结果附近进行精细搜索。
3. 根据导频相位估计网格平移。
4. 纠正几何变换后再执行 DCT 码字检测。

峰值支撑不足、最优和次优结果过于接近或置信度不足时，检测器会放弃同步结果，避免错误归因。

### 其他技术

| 技术 | 用途 |
| --- | --- |
| LSB | 在像素最低有效位中保存结构化载荷，适合无损场景和快速检测 |
| DCT / DWT / FFT | 传统频域水印和回退检测 |
| 小裁剪追踪 | 通过冗余和局部编码提高小范围裁剪后的检测概率 |
| 点阵追踪 | 将载荷分布到重复点阵中，辅助局部区域追踪 |
| 感知哈希 / 像素指纹 | 识别内容相同或近似的已登记图片 |
| ORB + RANSAC | 使用局部特征点和单应性估计进行视觉匹配和几何配准 |
| 可见版权水印 | 提供人工可见的版权提示和取证辅助信息 |

V4 启用时使用 V4 专用的 DCT + FFT 同步链路；小裁剪和点阵层主要用于非 V4 兼容模式，具体行为由接口参数和服务端配置决定。

## 技术栈

- Python 3.10+、FastAPI、Uvicorn、SQLAlchemy
- Pillow、OpenCV、NumPy、PyWavelets
- `reedsolo`、`python-multipart`、`python-dotenv`
- SQLite、MySQL/MariaDB、PostgreSQL
- pytest、HTTPX
- 可选：CuPy CUDA 12

## 项目结构

```text
main.py                         FastAPI 兼容入口
trace_app/
  api/                          HTTP 路由
  auth/                         登录、用户和角色权限
  database/                     数据库连接与仓储
  imaging/                      图片 IO、指纹、特征匹配和可见水印
  management/                   图片管理和仪表盘统计
  watermark/                    水印生成、提取和检测编排
  application.py                应用装配和生命周期
  config.py                     环境变量和运行参数
watermark_v4/                   V4 DCT、FFT、载荷和计算后端
frontend/                       前端代码和资源
tests/                          单元、API 和基准测试
tools/                          部署、迁移和辅助工具
README_DEPLOY.md                CentOS/RHEL 部署说明
```

## 本地安装

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`：

```dotenv
APP_NAME=BlindWatermarkTrace
ADMIN_USER=admin
ADMIN_PASS=change-this-password
DB_ENABLED=true
DB_URL=sqlite+pysqlite:///data/trace-dev.db
ROBUST_WATERMARK_VERSION=4
ROBUST_WATERMARK_STRENGTH=0.74
WATERMARK_AUTH_KEY=替换为至少32字节的随机密钥
TRACE_COMPUTE_DEVICE=auto
```

生成认证密钥：

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

启动：

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

访问 Web 页面：<http://127.0.0.1:8000>；Swagger UI：<http://127.0.0.1:8000/docs>。

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## API 用法

### 登录

```bash
curl -X POST http://127.0.0.1:8000/auth/login \
  -F "username=admin" \
  -F "password=change-this-password"
```

### 生成 V4 盲水印

```bash
curl -X POST http://127.0.0.1:8000/api/watermark/embed \
  -F "file=@./demo/1.jpg" \
  -F "user_id=demo-user" \
  -F "mode=dct" \
  -F "robust_watermark_version=4" \
  -F "robust_watermark_strength=0.74"
```

返回值包括 `trace_id`、状态、置信度以及带签名和有效期的原图/水印图访问地址。

### 提取和追踪

```bash
curl -X POST http://127.0.0.1:8000/api/watermark/extract \
  -F "file=@./uploads/watermarked/example.png"

curl -X POST http://127.0.0.1:8000/api/watermark/extract-url \
  -F "url=https://example.com/image.jpg"
```

主要接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/auth/login` | 用户登录 |
| POST | `/api/watermark/embed` | 嵌入盲水印并登记证据 |
| POST | `/api/watermark/extract` | 上传图片进行追踪 |
| POST | `/api/watermark/extract-url` | 下载 URL 图片并追踪 |
| GET | `/api/images` | 查询当前用户可见图片和统计 |
| DELETE | `/api/images/{image_id}` | 删除图片记录和关联文件 |
| GET | `/api/roles` | 查询角色和菜单权限 |
| GET/POST | `/api/users` | 查询或创建用户 |
| PUT/DELETE | `/api/users/{username}` | 修改或删除用户 |
| GET | `/api/dashboard-stats` | 查询仪表盘统计 |

完整参数以 `/docs` 中的 OpenAPI 定义和 `trace_app/api/` 路由为准。

## 重要配置

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `DB_URL` | SQLAlchemy 数据库连接串 | 空 |
| `DB_ENABLED` | 是否启用数据库 | `true` |
| `UPLOAD_DIR` | 原图和水印图目录 | `./uploads` |
| `DATA_DIR` | 数据和特征索引目录 | `./data` |
| `ROBUST_WATERMARK_VERSION` | 鲁棒水印版本 | `4` |
| `ROBUST_WATERMARK_STRENGTH` | 鲁棒水印强度 | `0.74` |
| `WATERMARK_AUTH_KEY` | V4 HMAC 密钥，至少 32 字节 | 空 |
| `TRACE_COMPUTE_DEVICE` | `auto`、`cpu` 或 `cuda` | `auto` |
| `WATERMARK_DETECTION_BUDGET_SECONDS` | 检测时间预算 | `5` |
| `ENABLE_SMALL_CROP_TRACE_REDUNDANCY` | 小裁剪冗余层 | `true` |
| `ENABLE_ALIGNED_AUTHENTICATED_DETECTION` | 配准认证检测 | `true` |

完整变量列表见 `.env.example`。修改 V4 的格式契约参数会导致已有 V4 图片无法解码，应同步升级水印版本。

## GPU 加速

具备 NVIDIA GPU 和 CUDA 12 环境时可选安装：

```powershell
python -m pip install -r requirements-gpu.txt
```

`TRACE_COMPUTE_DEVICE=auto` 会选择可用后端；`cpu` 强制 CPU 且不导入 CuPy；`cuda` 请求 CUDA，但设备不可用时仍会回退到 CPU。

## 测试

```powershell
python -m pytest -q
python -m pytest tests/test_watermark_v4_*.py -q
python -m pytest tests/test_application_structure.py::test_imaging_fingerprints_hashes_file_bytes_with_md5 tests/test_watermark_v4_api.py::test_v4_exact_file_fingerprints_succeed_without_image_decode -q
```

商业样本和攻击鲁棒性测试位于 `tests/`，部分测试需要额外图片数据集或模型文件。

## 部署

CentOS/RHEL、systemd、MySQL/MariaDB、PostgreSQL 迁移、发布包生成和 GPU 自适应安装请参阅 [README_DEPLOY.md](README_DEPLOY.md)。

生产部署前应设置随机的 `WATERMARK_AUTH_KEY`，使用最小权限数据库账号，配置安全的管理员密码，并通过 HTTPS 暴露登录和媒体下载接口。

## 安全和提交边界

请勿将以下内容提交到公开仓库：

- `.env`、数据库连接串、管理员密码和 `WATERMARK_AUTH_KEY`。
- `data/`、`uploads/`、数据库文件、真实业务图片和特征索引。
- 生产日志、调试日志、测试输出和发布压缩包，除非明确需要发布。

```powershell
git status --short
git check-ignore -v .env data uploads release
```

## 限制

- 盲水印不能保证对任意攻击、重绘、截图或大面积裁剪都可恢复。
- 检测结果应结合置信度、证据记录和人工审核使用，不应单独作为法律结论。
- 认证密钥丢失后，历史 V4 水印无法完成认证；更换密钥前应规划版本和数据迁移。
- 当前登录会话保存在进程内存中，服务重启后会话失效，多副本部署需要额外的会话存储或粘性会话。

## License

当前仓库未声明开源许可证。公开发布前请根据实际授权方式补充 `LICENSE` 文件。
