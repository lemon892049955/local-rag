"""命令行入口

提供终端下的快速入库和检索能力：
  python cli.py ingest <url>           - URL 入库
  python cli.py ingest-file <文件路径>  - 文件入库 (PDF/图片/音频)
  python cli.py search <query>         - 检索
  python cli.py reindex                - 重建索引
  python cli.py list                   - 列出所有知识文件
  python cli.py stats                  - 系统状态
  python cli.py wiki-compile           - 全量 Wiki 编译
  python cli.py wiki-inspect           - Wiki 健康检查
  python cli.py wiki-list              - 列出所有 Wiki 页面
"""

import asyncio
import sys
import json

from config import DATA_DIR


def print_json(data, indent=2):
    """美化打印 JSON"""
    print(json.dumps(data, ensure_ascii=False, indent=indent))


async def cmd_ingest(url: str):
    """入库命令"""
    print(f"🔗 URL: {url}")
    print("📥 正在入库...")

    from services.ingest_pipeline import ingest_url
    result = await ingest_url(url)

    if result.get("duplicate"):
        print(f"⚠️  该 URL 已入库")
        return

    if not result.get("success"):
        print(f"❌ 入库失败: {result.get('error', '未知错误')}")
        return

    print(f"   标题: {result.get('title', '未知')}")
    print(f"   标签: {', '.join(result.get('tags', []))}")
    print(f"   文件: {result.get('file_path', '')}")
    print(f"\n✅ 入库完成!")


async def cmd_ingest_file(file_path: str):
    """文件入库命令 (PDF/图片/音频)"""
    from pathlib import Path
    from ingestion.dispatcher import Dispatcher

    filepath = Path(file_path)
    if not filepath.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    dispatcher = Dispatcher()
    file_type = dispatcher.detect_type(str(filepath))
    print(f"📄 文件: {filepath.name} (类型: {file_type})")

    if file_type == "unknown":
        print(f"❌ 不支持的文件类型: {filepath.suffix}")
        return

    # 解析
    print("📥 正在解析文件...")
    try:
        raw = await dispatcher.dispatch(str(filepath))
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        return

    # 入库
    print("💾 正在入库...")
    from services.ingest_pipeline import ingest_raw
    result = await ingest_raw(raw, source_url=f"file://{filepath.name}")

    if not result.get("success"):
        print(f"❌ 入库失败: {result.get('error', '未知错误')}")
        return

    print(f"   标题: {result.get('title', '')}")
    print(f"   标签: {', '.join(result.get('tags', []))}")
    print(f"\n✅ 文件入库完成!")
    print(f"   文件: {saved_path}")
    print(f"\n✅ 文件入库完成!")


async def cmd_search(query: str, top_k: int = 3):
    """检索命令"""
    from main import get_searcher

    print(f"🔎 查询: {query}\n")

    searcher = get_searcher()
    result = await searcher.search(query=query, top_k=top_k)

    print("📝 回答:")
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)

    if result["sources"]:
        print(f"\n📚 参考来源 ({len(result['sources'])} 条):")
        for i, src in enumerate(result["sources"], 1):
            print(f"   [{i}] {src['title']} (距离: {src['distance']})")
            if src.get("source_url"):
                print(f"       URL: {src['source_url']}")


async def cmd_reindex():
    """重建索引"""
    from main import get_indexer

    print("🔄 正在重建向量索引...")
    total = get_indexer().reindex_all()
    print(f"✅ 完成! 共索引 {total} 个切片")


def cmd_list():
    """列出所有知识文件"""
    from storage.markdown_engine import MarkdownEngine

    engine = MarkdownEngine()
    files = engine.list_all()

    if not files:
        print("📭 知识库为空，请先通过 ingest 命令入库内容")
        return

    print(f"📚 知识库共 {len(files)} 篇:\n")
    for f in files:
        tags_str = ", ".join(f.get("tags", []))
        print(f"  📄 {f.get('title', '未知')}")
        print(f"     标签: {tags_str}")
        print(f"     摘要: {f.get('summary', '')}")
        print(f"     文件: {f.get('file_path', '')}")
        print()


def cmd_stats():
    """系统状态"""
    from main import get_indexer

    file_count = len(list(DATA_DIR.glob("*.md")))
    idx_stats = get_indexer().get_stats()

    print("📊 系统状态:")
    print(f"   知识文件数: {file_count}")
    print(f"   索引切片数: {idx_stats['total_chunks']}")


async def cmd_wiki_compile():
    """全量 Wiki 编译"""
    from wiki.compiler import WikiCompiler

    compiler = WikiCompiler()
    print("📝 开始全量 Wiki 编译...\n")

    count = 0
    for md_file in sorted(DATA_DIR.glob("*.md")):
        print(f"  编译: {md_file.name}")
        try:
            result = await compiler.compile(md_file)
            for detail in result.get("log_details", []):
                print(f"    {detail}")
            count += 1
        except Exception as e:
            print(f"    ❌ 失败: {e}")

    # 重建索引
    from wiki.index_builder import rebuild_index
    rebuild_index()

    print(f"\n✅ 编译完成! 处理了 {count} 篇文章")


def cmd_wiki_inspect():
    """Wiki 健康检查"""
    from wiki.inspector import inspect

    print("🔍 正在检查 Wiki 健康状态...\n")
    report = inspect()

    print(f"📊 {report['summary']}\n")

    if report["orphan_pages"]:
        print(f"  🔗 孤立页面 ({len(report['orphan_pages'])}):")
        for p in report["orphan_pages"]:
            print(f"    - {p}")

    if report["missing_pages"]:
        print(f"  ❓ 缺失引用 ({len(report['missing_pages'])}):")
        for p in report["missing_pages"]:
            print(f"    - [[{p}]]")

    if report["stale_pages"]:
        print(f"  ⏰ 过时页面 ({len(report['stale_pages'])}):")
        for p in report["stale_pages"]:
            print(f"    - {p['title']} (更新于 {p['updated_at']})")


def cmd_wiki_list():
    """列出所有 Wiki 页面"""
    from wiki.page_store import list_wiki_pages

    pages = list_wiki_pages()
    if not pages:
        print("📭 Wiki 为空，请先入库文章触发编译")
        return

    print(f"📝 Wiki 共 {len(pages)} 个页面:\n")
    for p in pages:
        sources_count = len(p.get("sources", []))
        print(f"  [{p.get('type', '')}] {p.get('title', '')}")
        print(f"    摘要: {p.get('summary', '')}")
        print(f"    来源: {sources_count} 篇 | 路径: {p.get('path', '')}")
        print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    if command == "ingest":
        if len(sys.argv) < 3:
            print("用法: python cli.py ingest <url>")
            return
        asyncio.run(cmd_ingest(sys.argv[2]))

    elif command == "ingest-file":
        if len(sys.argv) < 3:
            print("用法: python cli.py ingest-file <文件路径>")
            print("  支持: PDF (.pdf) / 图片 (.jpg .png .webp) / 音频 (.mp3 .m4a .wav)")
            return
        asyncio.run(cmd_ingest_file(sys.argv[2]))

    elif command == "search":
        if len(sys.argv) < 3:
            print("用法: python cli.py search <query> [top_k]")
            return
        query = sys.argv[2]
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        asyncio.run(cmd_search(query, top_k))

    elif command == "reindex":
        asyncio.run(cmd_reindex())

    elif command == "list":
        cmd_list()

    elif command == "stats":
        cmd_stats()

    elif command == "serve":
        import uvicorn
        from config import HOST, PORT
        print(f"🚀 启动服务: http://{HOST}:{PORT}")
        uvicorn.run("main:app", host=HOST, port=int(PORT), reload=True)

    elif command == "wiki-compile":
        asyncio.run(cmd_wiki_compile())

    elif command == "wiki-inspect":
        cmd_wiki_inspect()

    elif command == "wiki-list":
        cmd_wiki_list()

    else:
        print(f"未知命令: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
