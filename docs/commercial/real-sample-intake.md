# 真实传播样本采集规程

本规程用于采集可审计的真实传播证据。每个 `source_id`、`route` 和尝试序号构成一次独立采集；失败尝试也不得被后续尝试覆盖。

## 采集前准备

1. 仅使用批准的测试账号和批准的测试内容。确认内容不含真实用户资料或业务秘密。
2. 指定采集操作员 `operator`，并为发布证据指定一名未参与本次采集的复核员。
3. 在发送前确定 `source_id`、`route` 和三位尝试序号。尝试从 `001` 开始，重试时递增，禁止复用。
4. 在 `tests/fixtures/commercial/samples/real-platform/source/` 下准备不可变源文件，并将其设为只读或以其他方式防止改写。源文件原件和传播后文件都必须保留；传播后文件只保存在 `tests/fixtures/commercial/samples/real-platform/received/`。
5. 预先确定接收保存位置和文件名，确保接收动作不会要求事后重命名。所有清单和命令只使用该目录下的相对 POSIX 路径；禁止绝对路径，禁止密钥或其他秘密出现在路径、文件名、清单或 `notes` 中。
6. 记录设备、应用、浏览器或平台的名称和版本。校准系统时间，时间戳使用带时区的 RFC3339 格式。

## 路由操作步骤

每条路由都必须完整执行“发送/上传”和“接收/下载/截图”，不得以本机复制代替真实传播。

### wechat

1. 发送/上传：从批准的测试账号进入指定测试聊天或频道，以文件方式发送不可变源文件；不要使用会主动压缩图片的相册发送方式。立即记录 `sent_at`、发送账号/频道和发送端设备及应用名称和版本。
2. 接收/下载/截图：在独立接收端账号或设备中等待消息完成到达，记录 `received_at`。使用“另存为/下载原文件”直接保存到预先确定的 `output_relative_path`；如果测试对象明确是微信显示结果，则使用系统截图功能直接写入预定路径，并在 `notes` 说明截图范围、设备显示设置及原因。
3. 关闭预览，不使用编辑、转发或清理功能。立即执行哈希和证据记录。

### browser

1. 发送/上传：在指定浏览器和批准的测试站点中选择不可变源文件上传；等到上传成功状态出现后记录 `sent_at`、测试账号/频道、浏览器名称和版本以及站点/平台名称和版本。
2. 接收/下载/截图：等待服务端结果可用并记录 `received_at`。使用浏览器下载并在保存对话框中直接指定预定 `output_relative_path`；若证据要求网页渲染结果，则用系统或浏览器截图直接写入预定路径，并在 `notes` 记录页面、缩放比例和截图原因。
3. 不通过浏览器扩展、在线转换器或图片查看器重新保存文件。立即执行哈希和证据记录。

### target_platform

1. 发送/上传：登录批准的目标平台测试账号，选择不可变源文件，按正式用户流程提交到指定测试频道或任务；平台确认收到后记录 `sent_at`、账号/频道以及平台名称和版本。
2. 接收/下载/截图：等平台处理完成并记录 `received_at`。优先下载平台产物到预定 `output_relative_path`；若平台仅提供可视结果，则按测试协议截图并直接写入预定路径，在 `notes` 说明无法下载的原因、页面或任务标识和截图范围。
3. 保留平台成功页或任务标识的非敏感描述，不复制访问令牌或私密链接。立即执行哈希和证据记录。

## 文件保全与命名

- 不可变源文件与不可变接收文件是两份不同证物，禁止覆盖原件。采集全程禁止图像编辑、禁止重新压缩、禁止清理元数据。
- 接收后，在完成哈希和证据记录前禁止重命名，也不得用“另存为”重新编码。如果后续分析确需裁切、标注、转换或友好文件名，应从已记录证物复制并派生独立工作副本；工作副本不得替代证物。
- 接收证物的确定性命名格式为 `source_id--route--attempt-NNN.ext`，例如 `route-source-0001--wechat--attempt-001.png`。源证物必须命名为 `source_id.ext`，例如 `route-source-0001.png`。扩展名必须是实际图像格式，支持 `.png`、`.jpg`、`.jpeg`、`.webp`、`.bmp`、`.tif` 或 `.tiff`，不得仅修改扩展名伪装格式。
- 在接收或截图前就把确定性名称填入保存目标。若应用强制生成名称，保留该原生接收文件，先按原名计算哈希并记录，再制作字节不变的独立证据副本；不得移动或重命名原生接收文件，且须在 `notes` 同时记录两个相对路径及复制原因。

## SHA-256 计算与记录

对源文件和接收文件分别计算 SHA-256。以下示例对应清单中的 `route-source-0001`、`wechat`、`attempt: 1`；采集其他记录时，按同一命名规则替换为该记录的实际相对 POSIX 路径。不得在哈希前打开后保存文件。

PowerShell 精确命令：

```powershell
Get-FileHash -Algorithm SHA256 tests/fixtures/commercial/samples/real-platform/source/route-source-0001.png
Get-FileHash -Algorithm SHA256 tests/fixtures/commercial/samples/real-platform/received/route-source-0001--wechat--attempt-001.png
```

Python 精确命令：

```shell
python -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('tests/fixtures/commercial/samples/real-platform/source/route-source-0001.png').read_bytes()).hexdigest())"
python -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('tests/fixtures/commercial/samples/real-platform/received/route-source-0001--wechat--attempt-001.png').read_bytes()).hexdigest())"
```

四条命令必须全部运行。分别比较源文件的两次结果和接收文件的两次结果，PowerShell 与 Python 的结果必须一致。将一致的源文件结果记录为 `source_sha256`，将一致的接收文件结果记录为 `received_sha256`，使用完整的 64 位十六进制值，并记录实际输入路径。任一复算结果不一致，立即停止，该样本按拒收处理。

## 证据记录字段

每次尝试必须记录以下字段，不适用的环境字段写 `not_applicable`，不得留给操作员猜测：

| 字段 | 记录要求 |
| --- | --- |
| `operator` | 操作员的非敏感标识 |
| `sent_at` | 实际发送/上传完成时间，RFC3339 |
| `received_at` | 实际收到、下载可用或截图完成时间，RFC3339 |
| `route` | 仅为 `wechat`、`browser` 或 `target_platform` |
| `attempt` | 从 1 开始的正整数；文件名中写为三位数 |
| `source_id` | 与源清单一致的稳定标识 |
| `output_relative_path` | 接收证物的相对 POSIX 路径 |
| `source_relative_path` | 源证物的相对 POSIX 路径 |
| `source_sha256` | 源证物 SHA-256 |
| `received_sha256` | 接收证物 SHA-256 |
| `device` | 设备名称和版本 |
| `software` | 应用名称、浏览器名称或平台名称 |
| `software_version` | 应用名称和版本、浏览器名称和版本或平台名称和版本中的版本值 |
| `account_channel` | 账号/频道的非敏感测试标识；敏感时使用批准的别名 |
| `notes` | 异常、截图原因、平台任务标识和其他必要说明；不得包含秘密 |
| `reviewer` | 独立第二人的非敏感标识；必须由复核员本人填写，且不得等于 `operator` |
| `rejection_reason` | 拒收原因；正常待采集为空，失败或不确定的待采集记录必须非空 |

每次尝试复制以下完整 JSON 记录模板，并替换值；字段不得删除：

```json
{
  "source_id": "route-source-0001",
  "route": "wechat",
  "attempt": 1,
  "sent_at": null,
  "received_at": null,
  "source_relative_path": "tests/fixtures/commercial/samples/real-platform/source/route-source-0001.png",
  "output_relative_path": "tests/fixtures/commercial/samples/real-platform/received/route-source-0001--wechat--attempt-001.png",
  "source_sha256": null,
  "received_sha256": null,
  "status": "pending_collection",
  "operator": "",
  "device": "",
  "software": "",
  "software_version": "",
  "account_channel": "",
  "notes": "",
  "reviewer": "",
  "rejection_reason": ""
}
```

所有路径必须位于 `tests/fixtures/commercial/samples/real-platform/` 下。清单中只写仓库相对路径，不写盘符、UNC 路径、用户目录或外部 URL。

## 状态更新与拒收处理

只有以下条件全部满足，才可发布 `status: collected` 记录：源文件和接收文件均存在且能按扩展名解码；`received_at` 不早于 `sent_at`；`operator`、`account_channel` 已记录；`source_sha256` 与 `received_sha256` 已记录且与文件字节匹配；`device`、`software`、`software_version` 已记录；独立第二人已完成复核并由本人填写非空 `reviewer`，且在非空 `notes` 中记录复核时间和结论；`rejection_reason` 为空；其他必填证据字段已完成。

不得直接把官方清单改成已采集状态。按以下顺序从仓库根目录操作：

1. 从仍为待采集状态的官方清单创建工作清单副本：

```powershell
Copy-Item tests/fixtures/commercial/manifests/real-platform-routes.json tests/fixtures/commercial/manifests/real-platform-routes.working.json
```

2. 只编辑 `real-platform-routes.working.json`。在其中填写所有证据字段，把本次记录设为 `status: collected`，由第二人填写 `reviewer` 并在 `notes` 写入复核时间和结论。
3. 校验工作清单副本，不校验尚未更新的官方清单：

```shell
python -m tests.commercial_dataset_manifest tests/fixtures/commercial/manifests/real-platform-routes.working.json --kind routes --root .
```

4. 只有命令退出码为 0、没有校验错误且输出计数正确，才表示 `validator passes`。随后用同一文件系统上的原子替换发布已验证副本：

```shell
python -c "import os; os.replace('tests/fixtures/commercial/manifests/real-platform-routes.working.json', 'tests/fixtures/commercial/manifests/real-platform-routes.json')"
```

校验失败时不得执行上述原子替换，官方清单保持 `pending_collection`。删除或隔离失败的已采集工作副本，再从官方清单创建新的拒收工作副本：

```powershell
Copy-Item tests/fixtures/commercial/manifests/real-platform-routes.json tests/fixtures/commercial/manifests/real-platform-routes.rejection.working.json
```

在新的拒收工作副本中记录实际已出现的时间、操作者、哈希和路径；该记录的 `status` 仍为 `pending_collection`，并填写非空 `rejection_reason` 和非空 `notes`。然后校验并原子发布这份待采集拒收记录：

```shell
python -m tests.commercial_dataset_manifest tests/fixtures/commercial/manifests/real-platform-routes.rejection.working.json --kind routes --root .
python -c "import os; os.replace('tests/fixtures/commercial/manifests/real-platform-routes.rejection.working.json', 'tests/fixtures/commercial/manifests/real-platform-routes.json')"
```

文件缺失、文件损坏、发生编辑或重新编码、路由无法确认、时间或操作者不确定、哈希不一致的样本必须保持 `pending_collection`，并同时填写非空 `rejection_reason` 和非空 `notes` 说明拒收原因及观察事实。只要任一时间戳、操作者、任一哈希或任一证据文件已经出现，也适用此规则；只有完全未开始且文件不存在的初始槽位可以全部留空。清单只允许 `pending_collection` 和 `collected` 两种状态，不建立第三种拒收状态。此类样本不得计入 collected/pass/fail evidence，也不得用另一次尝试静默覆盖。

模拟样本不得标记为真实路由证据。模拟结果必须单独报告，与真实路由的数量、通过率、失败率和结论分开呈现。

## 证据保管链与复核清单

操作员按顺序勾选并签署以下保管链；每次文件接触或复制都在 `notes` 写明人员、时间、动作、输入路径、输出路径和哈希：

- [ ] 使用批准内容，源证物路径、只读状态和 `source_sha256` 已记录。
- [ ] 发送/上传路由、账号/频道、环境版本和 `sent_at` 已记录。
- [ ] 接收/下载/截图未经过编辑，`received_at` 和原始保存路径已记录。
- [ ] 接收证物保持不可变，`received_sha256` 已记录并复算一致。
- [ ] 命名映射到正确的 `source_id`、`route` 和尝试序号，所有路径均合规。
- [ ] 异常、复制和派生工作副本均可追溯，拒收样本未进入结果统计。
- [ ] validator passes 后才更新采集状态。

用于发布的证据必须完成独立第二人复核。复核员重新读取文件而非采用操作员口述，复算两个 SHA-256，核对时间顺序、路由、环境、路径、状态门槛和隐私要求，并由复核员本人把自己的非敏感标识记录到 `reviewer`，在 `notes` 追加复核时间和结论。`reviewer` 不得与 `operator` 相同。存在差异时不得发布，样本保持 `pending_collection` 并填写拒收原因。

## 隐私与安全

- 禁止凭据、令牌、聊天导出和私人个人内容进入样本、截图、文件名、清单、日志或报告。
- 只使用批准的测试账号与批准的测试内容；账号/频道仅记录非敏感标识或批准的别名。
- 截图前关闭无关窗口和通知，确认画面不含联系人、会话历史、私密 URL、访问令牌或设备个人信息。
- 发现敏感内容时不要编辑证物来遮盖；应停止采集、限制访问并按拒收流程重新使用合规内容采集。
