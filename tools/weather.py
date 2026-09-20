"""
天气查询工具
"""
import os
import requests
from tools.base import Tool


class WeatherTool(Tool):
    """天气查询工具，使用高德地图API，支持任意城市"""

    def __init__(self):
        super().__init__(
            name="weather",
            description="查询任意城市的天气，输入城市名称即可"
        )

    def execute(self, city: str) -> str:
        """查询天气 - 高德API支持直接传城市名"""
        try:
            resp = requests.get(
                "https://restapi.amap.com/v3/weather/weatherInfo",
                params={
                    "key": os.getenv("AMAP_MAPS_API_KEY", ""),
                    "city": city,
                    "extensions": "base"
                },
                timeout=10
            )
            data = resp.json()
            if data.get("status") == "1" and data.get("lives"):
                live = data["lives"][0]
                return f"{city}：{live['weather']}，温度{live['temperature']}°，{live['winddirection']}风{live['windpower']}级"
            return f"查询失败：{data.get('info', '未知错误')}"
        except Exception as e:
            return f"查询出错: {e}"

    def get_schema(self) -> dict:
        """返回参数schema"""
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如北京、上海"}
            },
            "required": ["city"]
        }
