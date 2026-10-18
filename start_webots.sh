#!/bin/bash
# 启动Webots并加载导航世界

WORLD_FILE="webots/worlds/navigation.wbt"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "无人系统智能导航 - Webots仿真启动"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"
echo "世界文件: $WORLD_FILE"
echo ""

# 检查世界文件是否存在
if [ ! -f "$PROJECT_DIR/$WORLD_FILE" ]; then
    echo "错误: 世界文件不存在: $PROJECT_DIR/$WORLD_FILE"
    exit 1
fi

# 寻找 Webots 可执行文件
WEBOTS_BIN=""
if command -v webots >/dev/null 2>&1; then
    WEBOTS_BIN="$(command -v webots)"
elif [ -f "/Applications/Webots.app/Contents/MacOS/webots" ]; then
    WEBOTS_BIN="/Applications/Webots.app/Contents/MacOS/webots"
elif [ -f "/usr/local/bin/webots" ]; then
    WEBOTS_BIN="/usr/local/bin/webots"
elif [ -f "/snap/bin/webots" ]; then
    WEBOTS_BIN="/snap/bin/webots"
fi

if [ -z "$WEBOTS_BIN" ]; then
    echo "提示: 未在系统路径检测到 Webots 可执行文件"
    echo "  - Linux: sudo snap install webots 或从 cyberbotics.com 下载"
    echo "  - macOS: brew install --cask webots"
    echo ""
    echo "本项目内置了 Mock 模拟传感器支持，无需物理运行 Webots 也可完整演示所有算法："
    echo "  python web/app.py"
    echo "  python scripts/demo/demo_webots.py"
    exit 0
fi

echo "正在启动 Webots: $WEBOTS_BIN ..."
"$WEBOTS_BIN" "$PROJECT_DIR/$WORLD_FILE" &

echo "Webots 已启动并正在加载场景 $WORLD_FILE"
echo ""
echo "请在终端中启动导航主程序:"
echo "  python web/app.py"