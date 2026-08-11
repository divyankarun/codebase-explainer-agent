import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from config import GITHUB_TOKEN


async def main():
    server_params = StdioServerParameters(
        command="docker",
        args=[
            "run",
            "-i",
            "--rm",
            "-e",
            f"GITHUB_PERSONAL_ACCESS_TOKEN={GITHUB_TOKEN}",
            "ghcr.io/github/github-mcp-server",
        ],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 1. List tools
            tools_res = await session.list_tools()
            print("--- AVAILABLE TOOLS ---")
            for t in tools_res.tools:
                print(f"Tool: {t.name}, Description: {t.description[:80]}...")
                if t.name in ["get_file_contents", "search_code"]:
                    print(f"  Schema for {t.name}: {json.dumps(t.input_schema, indent=2)}")


            # 2. Get file contents for README.md
            res = await session.call_tool("get_file_contents", arguments={"owner": "psf", "repo": "requests", "path": "README.md"})
            print("\n--- GET FILE CONTENTS (README.md) ---")
            print("Number of content blocks:", len(res.content))
            for i, c in enumerate(res.content):
                print(f"Block {i} type:", type(c), "repr:", repr(c))

if __name__ == "__main__":
    asyncio.run(main())
