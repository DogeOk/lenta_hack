import requests
import os
import logging
from datetime import date, timedelta

from model import forecast

URL_CATEGORIES = "categories"
URL_SALES = "sales"
URL_STORES = "shops"
URL_FORECAST = "forecast/"

api_port = os.environ.get("API_PORT", "8000")
api_host = os.environ.get("API_HOST", "host.docker.internal")

_logger = logging.getLogger(__name__)


def setup_logging():
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.DEBUG)
    handler_m = logging.StreamHandler()
    formatter_m = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
    handler_m.setFormatter(formatter_m)
    _logger.addHandler(handler_m)


def get_address(resource):
    return "http://" + api_host + ":" + api_port + "/api/v1/" + resource


def get_stores():
    stores_url = get_address(URL_STORES)
    resp = requests.get(stores_url)
    if resp.status_code != 200:
        _logger.warning("Could not get stores list")
        return []
    return resp.json()


def get_sales(store=None, sku=None):
    sale_url = get_address(URL_SALES)
    params = {}
    if store is not None:
        params["store"] = store
    if sku is not None:
        params["sku"] = sku
    resp = requests.get(sale_url, params=params)
    if resp.status_code != 200:
        _logger.warning("Could not get sales history")
        return []
    return resp.json()


def get_categs_info():
    categs_url = get_address(URL_CATEGORIES)
    resp = requests.get(categs_url)
    if resp.status_code != 200:
        _logger.warning("Could not get category info")
        return {}
    result = {el["sku"]: el for el in resp.json()}
    return result


def main(today=date.today()):
    forecast_dates = [today + timedelta(days=d) for d in range(1, 15)]
    forecast_dates = [el.strftime("%Y-%m-%d") for el in forecast_dates]
    categs_info = get_categs_info()
    for store in get_stores():
        for item in get_sales(store=store["store"]):
            item_info = categs_info[item["sku"]]
            sales = item["fact"]
            prediction = forecast(sales, item_info, store)
            post_response = {"store": store["store"],
                           "forecast_date": today.strftime("%Y-%m-%d"),
                           "forecast": {"sku": item["sku"],
                                        "sales_units": {k: v for k, v in zip(forecast_dates, prediction)}
                                        }
                           }
            requests.post(get_address(URL_FORECAST), json=post_response)


if __name__ == "__main__":
    setup_logging()
    main()
