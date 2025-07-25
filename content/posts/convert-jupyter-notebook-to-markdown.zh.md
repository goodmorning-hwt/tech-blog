+++
title = 'Convert Jupyter Notebook to Markdown'
date = 2025-07-25T09:54:47+08:00
draft = false
+++

将Jupyter Notebook（.ipynb文件）转换为Markdown（.md文件）的工具有很多，其中最常用和最官方的工具是 **nbconvert**。除了nbconvert之外，还有一些其他的命令行工具、VS Code插件和在线转换器可供选择。

### 1\. 使用官方工具 `nbconvert`

`nbconvert` 是一个由Jupyter社区开发的强大工具，可以将Jupyter Notebook转换为多种格式，包括Markdown、HTML、PDF、LaTeX等。它可以通过命令行或Python脚本来使用。

#### a. 命令行使用

这是最直接和常用的方法。首先，你需要确保已经安装了`nbconvert`。通常，如果你安装了Jupyter，`nbconvert`也会被一并安装。如果没有，可以通过pip进行安装：

```bash
pip install nbconvert
```

安装后，在你的终端或命令行中，进入到Jupyter Notebook文件所在的目录，然后运行以下命令：

```bash
jupyter nbconvert --to markdown your_notebook_name.ipynb
```

这条命令会生成一个名为 `your_notebook_name.md` 的Markdown文件。这个文件会包含你的Markdown单元格内容以及代码单元格和其输出。

**一些常用选项:**

  * **`--output-dir`**: 指定输出文件的目录。
  * **`--output`**: 指定输出文件的名称。
  * **`--no-input`**: 在输出中不包含代码单元格。

#### b. 在Jupyter Notebook/Lab中直接导出

你也可以在Jupyter Notebook或JupyterLab的界面中直接进行转换：

  * **在Jupyter Notebook中:**

    1.  打开你的 `.ipynb` 文件。
    2.  点击菜单栏的 "File" -\> "Download as"。
    3.  选择 "Markdown (.md)"。

  * **在JupyterLab中:**

    1.  打开你的 `.ipynb` 文件。
    2.  点击菜单栏的 "File" -\> "Export Notebook As..."。
    3.  选择 "Export to Markdown"。

#### c. 使用Python脚本

你还可以在Python脚本中调用`nbconvert`库来实现转换，这在自动化工作流程中非常有用。

```python
import nbformat
from nbconvert import MarkdownExporter

def convert_notebook_to_markdown(notebook_path, output_path):
    """
    将Jupyter Notebook转换为Markdown文件。

    :param notebook_path: 输入的.ipynb文件路径
    :param output_path: 输出的.md文件路径
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    markdown_exporter = MarkdownExporter()
    (body, resources) = markdown_exporter.from_notebook_node(nb)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(body)

# 使用示例
# convert_notebook_to_markdown('my_notebook.ipynb', 'my_notebook.md')
```

### 2\. 其他工具和插件

除了`nbconvert`，还有一些其他的选择：

  * **`ipynb-to-md`**: 一个简单的命令行工具，专门用于将Jupyter Notebook转换为Markdown。它注重于生成干净、易读的Markdown文件。
  * **Visual Studio Code 插件**: 如果你使用VS Code作为你的编辑器，可以安装一些插件来方便地进行转换，例如 "Jupyter To Markdown"。通常，你只需在`.ipynb`文件上右键点击，然后选择转换选项即可。
  * **在线转换器**: 也有一些网站提供在线的`.ipynb`到`.md`的转换服务。你只需上传你的Notebook文件，网站就会为你生成一个可以下载的Markdown文件。但请注意，对于包含敏感数据或代码的Notebook，使用在线工具可能存在安全风险。

### 总结

对于大多数用户来说，**`nbconvert`** 是将Jupyter Notebook转换为Markdown的首选工具，因为它功能强大、灵活且是官方支持的。无论是通过命令行、Jupyter界面还是Python脚本，它都能很好地满足你的需求。如果你需要一个更轻量级的命令行工具，或者更喜欢在VS Code中进行操作，那么`ipynb-to-md`和相关的VS Code插件也是不错的选择。