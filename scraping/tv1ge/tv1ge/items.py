from datetime import datetime

import scrapy

from itemloaders.processors import TakeFirst, MapCompose, Join


def parse_date(date_str):
    date_str = date_str.strip()
    date = datetime.strptime(date_str, '%H:%M, %d.%m.%Y')
    return date

def join_and_strip(values):
    joined_values = Join()(values)
    stripped_values = joined_values.strip()
    return stripped_values

class Tv1GeItem(scrapy.Item):
    date = scrapy.Field(
        input_processor=MapCompose(parse_date),
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
        input_processor=MapCompose(str.strip),
        output_processor=join_and_strip
    )
