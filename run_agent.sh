#!/bin/zsh
# 定时任务包装脚本：launchd 每天调用这个文件
# （launchd 不会自动加载 ~/.zshrc，所以这里手动加载以拿到API密钥）

source ~/.zshrc

cd /Users/jackietsang/quant-research-sandbox

echo ""
echo "════════════════════════════════════════════"
echo "  自动运行 $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════"

/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 trading_agent.py 2>&1

echo ""
echo "──── 当前持仓 ────"
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 trading_agent.py status 2>&1
