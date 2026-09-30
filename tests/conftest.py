"""测试共用的构造辅助。

放在 conftest 而不是各测试文件里，是为了让「怎么搭一张图」只有一处答案 ——
将来 `build_graph` 的签名变了（Task 11 / 14 会各加一个可选参数），
要改的也只有这里。
"""


def build_stub_graph(retriever, **kwargs):
    """用**桩检索**搭一张图。不碰项目 1、不碰数据库、不碰网络。"""
    from csagent.graph import build_graph

    return build_graph(retriever, **kwargs)
