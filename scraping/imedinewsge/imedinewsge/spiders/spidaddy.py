import json

import scrapy

from ..items import ImedinewsgeItem
from imedinewsge.utils import URL, get_next_url

from scrapy.loader import ItemLoader


class SpidaddySpider(scrapy.Spider):
    name = "spidaddy"
    allowed_domains = ["imedinews.ge"]
    start_urls = URL

    def parse(self, response):
        try:
            json_resp = json.loads(response.text)
            next_page_exists = json_resp['LoadMore']
            items = json_resp['List']

            for item in items:
                loader = ItemLoader(item=ImedinewsgeItem())
                loader.add_value('date', item['DateValue'])
                loader.add_value('title', item['Title'])
                loader.add_value('content_url', item['Url'])
                loader.add_value('content', item['Content'])

                yield loader.load_item()

            if next_page_exists:
                next_url = get_next_url(response.url)
                yield response.follow(next_url, callback=self.parse)

        except json.decoder.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON from {response.url}")
            next_url = get_next_url(response.url)
            yield response.follow(next_url, callback=self.parse)
