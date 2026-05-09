File is structured by directory and goes through several of the relevant files. Also I'm realizing that my scraping scripts are kinda all over the place. sorry

Hazelnut-insurance

Initially, I wanted to try and model the probability of my set of triggers then compare to some price action to get expected loss. This doesn’t work well because we don’t have true hazelnut price so we can’t get a corresponding change in price. This will be an overarching theme in the problems that I ran into. Additionally, this is when I did the first scrape of data coming from 1) ERA5 (weather), FAOSTAT (production), and try/usd.

At this point in time is when Aadvik and I had a conversation about trying to create the basket and we discussed my troubles with creating a basket because no other (liquid at least) nut futures exist. He suggested viewing similar equities to Ferrero. I quickly learned that despite other confectionary manufacturers were public, few of them use hazelnuts to the same extent. 

In terms of triggers, I did the research to determine what triggers actually have high impact. Here were my notes: 

Current triggers: 
- Spring frost (recency bias from claude perhaps)
- Heat/drought, changes oil content and makes kernel ratio drop. Imagine levant vs giggle nuts change
- Hail: weather again. Destroys nuts
- Production stat too low. TUIK production stat. This might be slow payout but definitely far more accurate. 
- TMO price floor action. There’s a floor on the price for nuts. This is a way Turkish gov regulates nut action. Saves farmers from getting crushed when yield is poor. Result of poor yield. Compare 2 year MA on this? 
    - Note that hazelnuts are on and off
- Disease
- Tariffs
- Supply disrupting events: not sure how to price this. This is kinda fucked no? Wild variance on payouts here? Like war in Indian Ocean should be impact less while war in mediterranean (like idk levant area bombign something idk)
- Some linear/tiered lira change payout? Probably linear the more I think about it. If lira depreciates, then hazelnut goes up in price. Bc fertilizer and tools are bought in dollars

Data

1. ERA5 data: weather related. 
    1. Frost calculation done by taking temperature data from the hazelnut strip and production weighting the temperatures as such. Note that we ignored other and renormalized the weights. 
        1. Ordu31.1%
        2. Giresun16.3%
        3. Samsun14.1%
        4. Sakarya12.8%
        5. Düzce10.7%
        6. Trabzon6.1%
        7. Other8.9%
2. Giresun spot prices scraped from giresun exchange website
3. TUIK hazelnut balance scraped from TUIK website/excel sheet. TUIK is the Turkish statistics institution or something
4. FAOSTAT is the production data of hazelnuts

Notebooks

I had to spend some time to find richer data. The giresun exchange data was a big help because it put TMO price limit in context and attached a price + volume to the hazelnuts at end of each month. A few hours I learned that giresun was a specific 

This directory I made some big leaps in the gaps of my knowledge. Hazelnut_insurance_pricing.ipynb, hazelnut_causality.ipynb, and hazelnut_pricing.ipynb was my first attempt to attach my triggers to production changes and come up with some pricing given the giresun pricing. This didn’t work that well and I concluded that frost was the most important factor. Many of the other factors lacked statistical significance or were not relevant logically. For example, TMO price should be baked into frost/weather event pricing because it would typically be a reaction to a devastating frost. 

One thing to note: there was a discrepancy in terms of the frosts because of a terrain factor. My frost calculations said there was a major frost in 2015, but after some research it happened in 2014. To better understand this, I had to look at a map and see where the provinces actually were. I learned that one of the hazelnut provinces, Duzce, was blowing up the frost metric while the 5 other relevant ones had zero frost notifications. This is due to the location of the Duzce sensor and the province itself. Duzce is not hidden by the mountains and is also lower in elevation causing cold air from heavy winds off the Black Sea to pool in the area. 

Most importantly, I had the price_production_regression.ipynb that was trying to bridge the gap between production values and the giresun prices. This leads me into the scripting. 

Scripts

Several scraping scripts to get the news and total giresun monthly data which I didn’t do at first. 

Important files: price_regression.py and production_regression.py. 

Ran an L1 regression to do feature selection and then L2 on those features. I threw in a bunch of features. The conclusion was that no equities or futures or assets were helpful in determining the giresun price. R^2 = 0.135

I did the same methodology for the production regression. The most important features ended up being the weather data. 

Modeling

Trying to solidify some of the modeling and feature selection in this directory. 

Price_models.ipynb: 

The most important models were those that involved production, which I designed to be a trailing shortfall over the past 5 years. This is working in yearly space. 

Production_models.ipynb: 

Production was very autoregressive. So AR(1) worked really well, but not very useful in this context because we’re trying to insure against the freak events. Frost was also very useful, but this should be an event trigger. I am able to get a model with R^2 = ~0.55-0.6 with this 

Event triggers: 

Some conclusions I made earlier that I did not cite. After looking at the European Severe Weather Report (ESWR), I saw that hail almost NEVER happened in the hazelnut strip. Off of this intuition, I scrapped hail as an event trigger. War and lira depreciation I was unsure how to model as an event trigger. Consider some NLP on some documents to forecast outcomes. 

I narrowed this down to two important event triggers that I actually had data for: frost and rain. I increased the depth of the frost and precipitation data do further back to just simply count instances that we had damaging degree hours. This notebook is doing the EDA to see frost outcomes and then how much production damage there was. Regressions here aren’t the best because you’d imagine that production fluctuates a lot on its own and the frost is impacting the most left tail scenarios. 

Similar mentality for rain. Something to note is that rain impacts hazelnuts from august to October during harvesting season because it causes mold. So that’s what I was testing.

Hazelnut_basket

I spent a few hours on day 3 trying to find better sources because the basket and the event triggers are nearly impossible without some true semblance of price. Since there doesn’t exist a real future on this, Turkey uses an electronically written receipts (EWR) to warehouse trades. I found a website for TOBB or some exchange regulator that tracks the EWR and provides physical delivery. They track the trades (which don’t happen every day), price, and volume. From this 

You can see the EDA of this data in hazelnut_basket_eda.ipynb

With this TOBB data, I had a better true hazelnut price. L1 for feature selection then L2. Ran regressions on annual data: 
1. Shortfall trailing 5 year production has ~0.3 R^2
2. Other features don’t speak much to it

TOBB data exists from 2005-2026. Note there are major gaps in this because there are times there are no trades at all

Then I did principal component regression: 
Removed production because we only get annual data. then ran regression on monthly. Garbage R^2. Then did monthly with principal component regression. Lots of collinearity in the features I selected. Need to spend some time choosing features, but not many liquid things for this. 

R^2 was good for PCA regression. 

However here are some issues: 
1. Lasso-->ridge method: fails because a ton of multicollinearity so lasso is zeroing a lot of features
2. PCA regression method: not bad gives 0.38 R^2

HOWEVER, there’s an issue when I run the same regression from just 2005-2020 the PCA regression dies (R^2 = 0.016). 

Issue here that needs to be solved.
