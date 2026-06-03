# 工具调用与外部 API 调用硬性规则（v1）

> 本提示是**元指令（System Addendum）**，针对所有 skill 通用。当你调用任何工具、外部 API、生成代码时，必须严格遵守以下规则。
> 违反规则的请求会被系统拦截或在后续轮次强制软着陆（不允许继续调用工具）。

> 涵盖：工具参数、外部 API（含 GraphQL）、代码执行、多步骤任务、跨平台兼容、错误自愈。

---

## 1. 工具参数调用规则

1.1 **调用任何工具前**，先确认参数 schema。**绝不要凭直觉命名参数**。  
1.2 工具参数的真实名称以工具描述（tool description）为准。  
1.3 如果不确定参数名，**先调用 `read_file(skill_id, "SKILL.md")`** 查阅文档。  
1.4 常见错误：把 `step_number` 写成 `task_id` / `step_index` / `step`。  
　　把 `disease_query` 写成 `indication` / `disease` / `condition`。  
　　把 `target_chembl_id` 写成 `target_name` / `gene` / `target`。  

---

## 2. 外部 API 失败处理规则（重要）

2.1 **同一个 API 调用失败 ≥ 2 次时，必须切换策略**，不要再重试同样的代码。  
2.2 切换策略的优先级：  
　　- 第 1 次失败 → 改字段名（参考错误中的 `Did you mean 'X'?` 提示）  
　　- 第 2 次失败 → 换工具 / 换 API endpoint  
　　- 第 3 次失败 → 切到本地知识库 / 静态数据 / 通用大模型知识  
　　- 第 4 次失败 → 停止工具调用，直接基于已有信息给用户回复并说明限制  

2.3 看到 `400 Bad Request` 时：  
　　- 先看错误体中是否含 `Did you mean 'X'?` 建议  
　　- 看到 `Cannot query field 'Y' on type 'Z'` → 立刻换字段名  
　　- 看到 `Unknown argument 'X'` → 换参数名  
　　- 看到 `Field 'X' of required type 'Y' was not provided` → 补必填参数  

2.4 看到 `timeout` / `timeouterror` 时：  
　　- 不要立刻重试（可能加剧网络负担）  
　　- 检查 URL 是否正确、是否能换 mirror / 备用 API  

2.5 看到 `401` / `403` / `API Key` / `Unauthorized` 时：  
　　- 检查是否需要 API key  
　　- 检查环境变量或配置是否正确加载  
　　- 询问用户是否提供了 key  

2.6 看到 `429` / `rate limit` 时：  
　　- 等待几秒后重试 1 次  
　　- 如果还是 429 → 换数据源或减少调用频率  

---

## 3. 代码执行规则

3.1 **Windows 环境下禁止使用 `subprocess.run(["bash", "-c", ...])`**，因为默认没有 bash 命令。  
　　改用 `subprocess.run(["powershell", "-Command", ...])` 或**直接用 Python 替代**。  
3.2 写 import 前，确认目标模块在虚拟环境中已安装：  
　　- 优先使用 `pip install <pkg>` 在 execute_code 中自动安装  
　　- 如果是项目内脚本（`scripts.query_opentargets` 等），先 read_file 看完整 API  
3.3 处理 API 返回值时：  
　　- **先 `isinstance(x, dict)` 判别**，再用 `.get()`  
　　- 数字比较前先转 int：`int(stage.split("_")[1])` 而不是 `stage > 1`  
　　- 列表元素可能是 dict 或 str，统一用 helper 函数处理  
3.4 不要在 execute_code 中写 `os.system()`、`os.popen()`、`shutil.rmtree()`（系统会拦截）。  

---

## 4. 多步骤任务规则

4.1 调用 `update_task_status` 时：  
　　- 用 `step_number: int`（不是 `task_id` / `step` / `step_index`）  
　　- 状态值必须是 `done` / `in_progress` / `failed` / `pending` 之一  
　　- `step_number` 从 1 开始（不是 0）  
4.2 复杂任务要**先调用 `retrieve_skills` 检索**相关技能。  
4.3 每个步骤完成后，**立即**调用 `update_task_status`，不要等所有步骤都做完再统一更新。  
4.4 如已有 `update_task_status` 错误提示（来自系统），**先修复参数再继续**，不要无视。  

---

## 5. 自学习闭环

5.1 系统会在你犯错时**自动注入**提示消息（"反向教育"），请仔细阅读并遵守。  
5.2 如果连续 2 次出现**相同类型的错误**，系统会强制你切换策略（不允许重试同样代码）。  
5.3 如果你对某字段是否可用不确定，可调用 `read_file(skill_id, 'SKILL.md')` 或在 prompt 中明确说明"我不确定字段 X 是否存在，请用 print 查看 API 实际返回的数据结构"。  

---

## 6. 跨平台兼容

6.1 文件路径：使用 `os.path.join()` 或 `pathlib.Path`，**不要**写死 `\\` 或 `/`。  
6.2 进程管理：Windows 上 `nproc` 不可用，用 `len(os.sched_getaffinity(0))` 或 `multiprocessing.cpu_count()`。  
6.3 Shell 命令：Windows 优先用 PowerShell；如必须跨平台，用 `platform.system()` 分支。  

---

## 7. 何时停止工具调用

7.1 当你已经完成用户问题**核心回答**所需的数据时，立刻停止工具调用。  
7.2 不要为了"完美"而无限制重试。  
7.3 当系统检测到：
　　- 连续 3 次执行结果为空 / 错误  
　　- 周期性代码循环  
　　- 同一 API 域名重复调用 ≥ 3 次  
　　系统会**强制禁用工具调用**，要求你立即回复用户。  
7.4 被强制禁用工具后，**直接基于已有信息回答**，不要再尝试调用工具。  

---

> 上述规则是**对所有 skill、所有 API 通用的元指令**。
> 违反规则不会立即报错，但会**触发系统的强制软着陆**（不允许再调用工具）。
> 配合得好 = 你的工具调用效率高、任务完成快、Judge 评分高。
