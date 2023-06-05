import scrapy

from ..utils import URLS, get_next_url, content_from_quote
from ..items import FormulanewsgeItem

from scrapy.loader import ItemLoader


class SpidaddySpider(scrapy.Spider):
    name = "spidaddy"
    allowed_domains = ["formulanews.ge"]
    start_urls = URLS

    def parse(self, response):
        items = response.xpath("//div[@class='col-lg-3 news__box__card']")
        for item in items:
            url = item.xpath(".//div[contains(@class,'date')]/following-sibling::a/@href").get()
            is_quote = item.xpath("./div[@class='main__phrases__box']")

            if is_quote:
                yield response.follow(url=url, callback=self.parse_quote)
            else:
                yield response.follow(url=url, callback=self.parse_content)

        next_page_exist = response.xpath("//body/*")
        if next_page_exist:
            next_url = get_next_url(response.url)
            yield response.follow(next_url, callback=self.parse)

    def parse_content(self, response):
        loader = ItemLoader(item=FormulanewsgeItem(), selector=response, response=response)
        loader.add_xpath('date', "//*[@class='news__inner__images_created']")
        loader.add_xpath('title', "//*[@class='news__inner__desc__title']/text()")
        loader.add_value('content_url', response.url)
        loader.add_xpath('content', "//section[@class='article-content']//text()")

        yield loader.load_item()

    def parse_quote(self, response):

        loader = ItemLoader(item=FormulanewsgeItem(), selector=response, response=response)
        loader.add_xpath('date', "//div[@class='phrase-date']")
        loader.add_xpath('title', "//div[@class='phrase-title']/text()")
        loader.add_value('content_url', response.url)
        quote = response.xpath("//div[@id='phrase-main']//text()").getall()
        quote_description = response.xpath("//div[@class='phrase-text'][2]//text()").getall()

        content = content_from_quote(quote, quote_description)
        loader.add_value('content', content)

        yield loader.load_item()
