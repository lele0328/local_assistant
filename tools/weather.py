"""
天气查询工具
"""
import requests
from tools.base import Tool


class WeatherTool(Tool):
    """天气查询工具，使用高德地图API"""

    CITY_CODES = {
        "北京": "110000", "上海": "310000", "广州": "440100",
        "深圳": "440300", "杭州": "330100", "成都": "510100",
        "武汉": "420100", "南京": "320100", "西安": "610100",
        "重庆": "500000",
    }

    def __init__(self):
        super().__init__(
            name="weather",
            description="查询城市天气，支持主要城市"
        )

    def execute(self, city: str) -> str:
        """查询天气"""
        if city not in self.CITY_CODES:
            return "不支持该城市"
        city_code = self.CITY_CODES[city]
        try:
            resp = requests.get(
                "https://restapi.amap.com/v3/weather/weatherInfo",
                params={
                    "key": os.getenv("AMAP_MAPS_API_KEY", ""),
                    "city": city_code,
                    "extensions": "base"
                },
                timeout=10
            )
            data = resp.json()
            if "lives" in data and data["lives"]:
                live = data["lives"][0]
                return f"{city}：{live['weather']}，温度{live['temperature']}°"
            return "查询失败"
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
