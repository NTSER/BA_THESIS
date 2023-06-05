from datetime import datetime
import re

import scrapy
from itemloaders.processors import TakeFirst, MapCompose

def timestamp_to_datetime(s):
    match = re.search(r"\d+", s)
    timestamp = int(match.group())
    try:
        date = datetime.fromtimestamp(timestamp / 1000)
    except:
        date = None
    return date

def html_to_text(html_string):
    selector = scrapy.Selector(text=html_string)
    text_list = selector.xpath("//p/text()").getall()
    text = '\n'.join(text_list)
    text = text.strip()

    return text

class ImedinewsgeItem(scrapy.Item):
    date = scrapy.Field(
        input_processor=MapCompose(timestamp_to_datetime),
        output_processor=TakeFirst()
    )
    title = scrapy.Field(
        input_processor=MapCompose(str.strip),
        output_processor=TakeFirst()
    )
    content_url = scrapy.Field(
        output_processor=TakeFirst()
    )
    content = scrapy.Field(
        input_processor=MapCompose(html_to_text),
        output_processor=TakeFirst()
    )
