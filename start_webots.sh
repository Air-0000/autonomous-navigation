#!/bin/bash
# 启动Webots并加载导航世界

WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD_FILE="webots/worlds/navigation.wbt"

# 获取项目根目录
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "无人系统智能导航 - Webots仿真启动"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"
echo "世界文件: $WORLD_FILE"
echo ""

# 检查Webots是否安装
if [ ! -f "$WEBOTS" ]; then
    echo "错误: Webots未安装"
    echo "请先安装Webots: brew install --cask webots"
    exit 1
fi

# 检查世界文件是否存在
if [ ! -f "$PROJECT_DIR/$WORLD_FILE" ]; then
    echo "错误: 世界文件不存在: $PROJECT_DIR/$WORLD_FILE"
    exit 1
fi

echo "正在启动Webots..."
echo ""

# 直接运行（Webots会打开窗口让用户手动加载世界）
open -a Webots

echo "Webots已打开!"
echo ""
echo "请按以下步骤操作:"
echo "1. 在Webots菜单中选择 File -> Open World"
echo "2. 浏览到: $PROJECT_DIR/$WORLD_FILE"
echo "3. 选中navigation.wbt并点击打开"
echo "4. 点击 'Play' ▶️ 按钮开始仿真"
echo ""
echo "然后在另一个终端运行主程序:"
echo "  cd $PROJECT_DIR"
echo "  python web/app.py"