
import os
import requests
from dotenv import load_dotenv

load_dotenv()


def get_coordinates(destination: str) -> dict:
    """여행지 이름을 위도와 경도로 변환한다."""

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY가 설정되지 않았습니다.")

    url = "https://maps.googleapis.com/maps/api/geocode/json"

    response = requests.get(
        url,
        params={
            "address": destination,
            "key": api_key,
        },
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "OK" or not data.get("results"):
        raise ValueError(
            f"좌표 조회 실패: {data.get('status')} "
            f"{data.get('error_message', '')}"
        )

    location = data["results"][0]["geometry"]["location"]

    return {
        "latitude": location["lat"],
        "longitude": location["lng"],
    }


def get_daily_forecast(destination: str, days: int = 3) -> list[dict]:
    """여행지의 일별 날씨 예보를 조회한다."""

    if not 1 <= days <= 10:
        raise ValueError("예보 조회 일수는 1~10일이어야 합니다.")

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY가 설정되지 않았습니다.")

    # 1. 여행지 이름을 좌표로 변환
    coordinates = get_coordinates(destination)

    # 2. Google Weather API 요청
    url = "https://weather.googleapis.com/v1/forecast/days:lookup"

    response = requests.get(
        url,
        params={
            "key": api_key,
            "location.latitude": coordinates["latitude"],
            "location.longitude": coordinates["longitude"],
            "days": days,
            "unitsSystem": "METRIC",
            "languageCode": "ko",
        },
        timeout=15,
    )
    response.raise_for_status()

    data = response.json()
    forecasts = []

    # 3. 에이전트에서 사용할 형식으로 변환
    for day in data.get("forecastDays", []):
        display_date = day["displayDate"]

        forecast_date = (
            f"{display_date['year']:04d}-"
            f"{display_date['month']:02d}-"
            f"{display_date['day']:02d}"
        )

        full_day = day.get("fullDayForecast", {})
        condition = full_day.get("weatherCondition", {})
        description = condition.get("description", {}).get("text")

        forecasts.append({
            "date": forecast_date,
            "condition": description,
            "temp_min_c": day.get(
                "minTemperature", {}
            ).get("degrees"),
            "temp_max_c": day.get(
                "maxTemperature", {}
            ).get("degrees"),
        })

    return forecasts


from datetime import date, timedelta


def get_trip_weather(trip: dict) -> list[dict]:
    """여행 날짜에 해당하는 날씨 예보를 수집한다."""

    start = date.fromisoformat(trip["start_date"])
    end = date.fromisoformat(trip["end_date"])

    if end < start:
        raise ValueError("여행 종료일이 시작일보다 빠릅니다.")

    # 여행 전체 기간 (당일 포함)
    trip_days = (end - start).days + 1

    # Google Weather API는 최대 10일 예보 제공
    forecast = get_daily_forecast(
        trip["destination"],
        days=min(trip_days + 10, 10)
    )

    # 실제 여행 날짜에 해당하는 예보만 반환
    return [
        day for day in forecast
        if start <= date.fromisoformat(day["date"]) <= end
    ]
