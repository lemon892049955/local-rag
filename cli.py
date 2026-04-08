"""命令行入口

提供终端下的快速入库和检索能力：
  python cli.py ingest <url>     - 入库
  python cli.py search <query>   - 检索
  python cli.py reindex          - 重建索引
  python cli.py list             - 列出所有知识文件
  python cli.py stats            - 系统状态
  python cli.py wiki-compile     - 全量 Wiki 编译（存量迁移）
  python cli.py wiki-inspect     - Wiki 健康检查
  python cli.py wiki-list        - 列出所有 Wiki 页面
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
    from ingestion.router import FetcherRouter
    from transform.llm_cleaner import LLMCleaner
    from storage.markdown_engine import MarkdownEngine
    from retrieval.indexer import VectorIndexer
    from utils.url_utils import check_duplicate

    print(f"🔗 URL: {url}")

    # 去重检查
    existing = check_duplicate(url, DATA_DIR)
    if existing:
        print(f"⚠️  该 URL 已入库: {existing}")
        return

    # 抓取
    print("📥 正在抓取内容...")
    router = FetcherRouter()
    try:
        raw = await router.fetch(url)
    except Exception as e:
        print(f"❌ 抓取失败: {e}")
        return

    print(f"   标题: {raw.title}")
    print(f"   来源: {raw.source_platform}")
    print(f"   内容长度: {len(raw.content)} 字符")

    # LLM 清洗
    print("🧹 正在 LLM 清洗...")
    cleaner = LLMCleaner()
    try:
        knowledge = await cleaner.clean(
            title=raw.title,
            content=raw.content,
            source=raw.source_platform,
            author=raw.author,
            original_tags=raw.original_tags,
        )
    except Exception as e:
        print(f"❌ LLM 清洗失败: {e}")
        return

    print(f"   清洗后标题: {knowledge.title}")
    print(f"   摘要: {knowledge.summary}")
    print(f"   标签: {knowledge.tags}")

    # 落库
    print("💾 正在落库...")
    engine = MarkdownEngine()
    filepath = engine.save(
        knowledge=knowledge,
        source_url=url,
        source_platform=raw.source_platform,
        author=raw.author,
    )
    print(f"   文件: {filepath}")

    # 索引
    print("🔍 正在建立向量索引...")
    indexer = VectorIndexer()
    chunk_count = indexer.index_file(filepath)
    print(f"   索引切片数: {chunk_count}")

    print(f"\n✅ 入库完成!")


async def cmd_search(query: str, top_k: int = 3):
    """检索命令"""
    from retrieval.searcher import RAGSearcher

    print(f"🔎 查询: {query}\n")

    searcher = RAGSearcher()
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
    from retrieval.indexer import VectorIndexer

    print("🔄 正在重建向量索引...")
    indexer = VectorIndexer()
    total = indexer.reindex_all()
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
    from retrieval.indexer import VectorIndexer

    file_count = len(list(DATA_DIR.glob("*.md")))
    indexer = VectorIndexer()
    idx_stats = indexer.get_stats()

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

    elif command == "ingest-text":
        if len(sys.argv) < 3:
            print("用法: python cli.py ingest-text <文本文件路径>")
            print("  文件第一行作为标题，其余作为正文")
            return
        asyncio.run(cmd_ingest_text(sys.argv[2]))

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
