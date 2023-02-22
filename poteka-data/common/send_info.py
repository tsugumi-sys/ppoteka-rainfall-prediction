import requests
import os
from logging import getLogger, INFO
from typing import Optional

logger = getLogger(__name__)
logger.setLevel(INFO)


def send_notify(msg: str) -> Optional[int]:
    """
    Replace your prefer notification services.
    This is LINE notify Example.
    """
    line_notify_token = os.getenv("LINE_TOKEN")

    if line_notify_token is None:
        logger.info("FAILED to get LINE_TOKEN from env.")
        return

    line_notify_endpoint = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_notify_token}"}
    data = {"message": msg}
    res = requests.post(url=line_notify_endpoint, headers=headers, data=data)

    return res.status_code
