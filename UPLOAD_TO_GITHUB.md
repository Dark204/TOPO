# 上传到 GitHub

先建 **Private（私有）仓库**。本地材料已经整理好；这一步不会公开给学生或其他人。确认论文、署名和代码许可后，再决定何时转为公开。

## 1. 建立仓库

1. 登录 https://github.com ，右上角点击 `+` → `New repository`。
2. 名称填写 `TOPO`，选择 `Private`。
3. 不勾选自动生成 README、.gitignore 或 License，本地已有前两项。
4. 点击 `Create repository`。

## 2. 上传代码

1. 解压本目录的 `release_assets/GitHub_Source.zip`。
2. 在新仓库页面点击 `uploading an existing file`，或 `Add file` → `Upload files`。
3. 将解压后的内容上传到仓库根目录：`README.md` 应直接显示在根目录，而不是再套一层 TOPO 文件夹。
4. 上传说明可写 `Add TopoTrace-AD code and experiment inputs`，点击 `Commit changes`。

上传解压后的代码文件，不是把 GitHub_Source.zip 当成代码文件上传。不拖整个本地 TOPO 目录：其中包含大数据和本地下载的 CARE 作者代码，均已从代码 ZIP 中排除。

## 3. 上传数据附件

进入仓库右侧 `Releases` → `Create a new release`（已有 release 时为 `Draft a new release`）。创建标签 `v1.0-data`，标题填写 `Experimental datasets and inputs`。

在附件区域放入以下五个文件：

- `datasets/TORAI_Main_270.zip`
- `datasets/RE2_OB_Raw_90.zip`
- `datasets/RE2_TT_Raw_90.zip`
- `datasets/RE3_OB_Generalization_30.zip`
- `release_assets/Experiment_Inputs.zip`

可再附上 `release_assets/checksums.json`。等所有附件上传完再点击 `Publish release`；它仍受私有仓库访问权限控制。只上传了一部分时使用 `Save draft`。

说明可填写：

> Source datasets and prepared supplementary inputs for TopoTrace-AD. The four source archives belong under datasets/. Extract Experiment_Inputs.zip into the repository root. Data sources and attribution are documented in datasets/README.md.

## 4. 检查

- 首页能直接看到 README、code、data、datasets 和 experiments。
- Releases 中有五个数据附件。
- 没有 results、日志、任务卡、历史版本、虚拟环境或论文写作材料。
- README 的运行命令与目录一致。数据 ZIP 不需要重新压缩。
- 正式公开前补充作者确定的代码 LICENSE 和论文引用。现有第三方数据署名不要删除。

网页上传普通文件的单文件上限是 25 MiB，普通 Git 文件上限是 100 MiB，所以本包把大数据独立为 Release 附件。代码 ZIP 里的文件均按网页上传大小检查。

官方说明：

- https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
- https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
- https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
