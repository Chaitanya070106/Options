from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from nsepython import nse_optionchain_scrapper
from tvDatafeed import TvDatafeed, Interval
from scipy.optimize import brentq
import time
from math import sqrt,log,exp,isnan, isinf
from scipy.stats import norm   # for normal dist.

## Replace 'your_username' and 'your_password' with TV credentials

username = 'princechaitanya07@gmail.com'
password = 'Tradingview*0701'

tv = TvDatafeed(username=username, password=password)

class stocks:
    def __init__(self,symbol,exchange,underlyingname,yfinance_ticker_name):
        self.symbol=symbol
        self.exchange=exchange
        self.underlying_name=underlyingname
        self.underlying_price=0
        self.min_strike=10000000000
        self.max_strike=0
        self.yf=yfinance_ticker_name
        self.expiries=[]
        self.strikes=[]
        self.optionstrings=[]
        self.risk_free_india=0.063
        self.option_data=[]
    def get_avalible_Strikes_and_expiry(self):
        symbol = "NIFTY"
        data = nse_optionchain_scrapper(symbol)
        count=0
        temp=None
        for item in data['records']['data']:
            
            self.expiries.append(item["expiryDate"])
            self.strikes.append(item["strikePrice"])

        for i in range(0,len(self.expiries)):
            self.expiries[i]=self.expiries[i].replace("-","")
            self.expiries[i]=datetime.strptime(self.expiries[i],"%d%b%Y")
            

        

    def Getting_strike_range(self):
        self.get_avalible_Strikes_and_expiry()
        self.underlying_price=tv.get_hist(symbol=self.underlying_name, exchange=self.exchange)
        print(self.underlying_price,"   ",self.underlying_name)
      
        self.underlying_price=self.underlying_price["close"].iloc[-1]
        #self.underlying_price=tv.get_price(self.underlying_name,self.exchange)["last"]

        #self.min_strike=self.underlying_price-self.underlying_price*0.30
        #self.max_strike=self.underlying_price+self.underlying_price*0.30
        strikes_temp=[]
        expiry_temp=[]
        today = datetime.today()
        #future_date = today + timedelta(days=30)
        future_date = datetime(2025, 7, 31)
        for i in range(0,len(self.strikes)):
            if (self.expiries[i]==future_date):
                strikes_temp.append(self.strikes[i])
                expiry_temp.append(self.expiries[i].strftime("%y%m%d"))
        self.strikes=strikes_temp
        self.expiries=expiry_temp
       

    def setting_option_strings(self):
        self.Getting_strike_range()
        for i in range(0,len(self.expiries)):
            self.optionstrings.append(self.underlying_name+str(self.expiries[i])+"C"+str(self.strikes[i]))
            #self.optionstrings.append(self.underlying_name+str(self.expiries[i])+"P"+str(self.strikes[i]))
        
    def get_option_pricies(self):
        self.setting_option_strings()
        df1 = tv.get_hist(symbol = "NIFTY", exchange = 'NSE' )
        Current_underlying=df1["close"].iloc[-1]
        for i in range(0,len(self.optionstrings)):
            df = tv.get_hist(symbol = self.optionstrings[i], exchange = 'NSE')
            time.sleep(1)
            strike=self.strikes[i]
            date=str(self.expiries[i])
            given_date_time=datetime(2000+int(date[0:2]),int(date[2:4]),int(date[4:6]),15,30)
            now=datetime.now()
            time_to_exp=given_date_time-now
            time_to_exp=time_to_exp.total_seconds()/(365*24*3600)
            model,val,actual_price=self.finding_implied_val(df,Current_underlying,strike,time_to_exp)
            print("time to exp in years is",time_to_exp)
            print("the implied val for ",self.optionstrings[i]," is ",val,"model price is",model)

            self.option_data.append({
                "STRIKE":self.strikes[i],
                "Expiation":self.expiries[i],
                "IV":val,
                "Model_price":model,
                "Actual_price":actual_price
            })
        df_output=pd.DataFrame(self.option_data)
        df_output.to_csv("Nifty_calc.csv",index=False)
        print("CSV file saved as 'nifty_option_pricing.csv'")


    def intial_vol_guess(self,C,S,T):
        return C/(S*(0,4*sqrt(T)))      # Brenner-Subrahmanyam approximation  this is used to approximate the val of a option. Work best for ATMs but not for other beacuse it does not take into account how the val chnages with respect to the strike.

    def d(self,sigma,S,K,r,t):
        
        d1=(log(S/K)+(r+(sigma**2)/2)*t)/(sigma*sqrt(t))
        d2=d1-sigma*sqrt(t)
        return d1,d2
    
    def call_price(self,S,K,r,t,d1,d2):
        C=norm.cdf(d1)*S-norm.cdf(d2)*K*exp(-r*t)
        return C
    

    def Val_from_price(self,S,K,r,t,c0):
       
        tol=1e-7
        epsilon= 1
        count=0
        max_iter=10000
        val=0.80
        while epsilon>tol:
            count+=1
            if count>=max_iter:
                print("breaking due to max count")
                break
            orig_val=val
            
            d1,d2=self.d(val,S,K,r,t)
            #newton raphsons method
            function_val=self.call_price(S,K,r,t,d1,d2)-c0
            
            vega=S*exp(0.5*d1**2)*sqrt(t)/(sqrt(2*3.14))
        
            try:
                val = val - function_val/vega  
            except:
                print("the val cal has failed. the vega value is ",vega)
            epsilon=abs((val-orig_val)/orig_val)    
        return val
    

    def finding_implied_val(self,data,nifty_current,strike,time_to_expiry):
        try:
            last_close=data["close"].iloc[-1]
            print("last close is", last_close)
            Val= self.Val_from_price(int(nifty_current), int(strike), self.risk_free_india, time_to_expiry, int(last_close))
            model=self.Finding_u_d_Binomial(nifty_current,strike,Val,time_to_expiry,self.risk_free_india)
            return model,Val,last_close
        except:
            return "-","-","-"

    def Finding_u_d_Binomial(self,underlying_spot,strike,vol,time_to_expiry,risk_free_rate):
        N=50
  #upto 5 times there is branching happeneing in the binonmial tree
        t=time_to_expiry/(N-1)
        u=exp(vol*sqrt(t))
        d=exp(-vol*sqrt(t))
        p=(exp(risk_free_rate*t)-d)/(u-d)
        stock_prices=np.zeros((N,N))
        call_prices=np.zeros((N,N)) #number of branches
        stock_prices[0,0]=underlying_spot
        for i in range(1,N):
            M=i+1
            stock_prices[i,0]=d*stock_prices[i-1,0]
            for j in range(1,M):
                stock_prices[i,j]=u*stock_prices[i-1,j-1]
        expiration = stock_prices[-1,:] - strike
        expiration.shape=expiration.size
        expiration=np.where(expiration>=0,expiration,0)
        call_prices[-1,:]=expiration
        for i in range(N-2,-1,-1):
            for j in range (i+1):
                call_prices[i,j]=exp(-risk_free_rate*t)*((1-p)*call_prices[i+1,j]+p*call_prices[i+1,j+1])
        return call_prices[0,0]



sto=stocks("NIFTY","NSE","NIFTY","NIFTY.NS")
sto.get_option_pricies()

"""
data = tv.get_hist(symbol='NIFTY', exchange='NSE', interval=Interval.in_1_minute, n_bars=1)
nifty_price=data['close'].iloc[-1]

print(nifty_price)
symbol_to_use="NIFTY250717C25150"
df = tv.get_hist(symbol = "NIFTY250717C25150", exchange = 'NSE', 
                 interval= Interval.in_1_minute, n_bars= 5000)
print(df)
print(df)



 delta=(-Upper_pay_off+lower_pay_off)/(d-u)
        Portfolio_Value=-Upper_pay_off+(u)*delta
    
        print("portfolio value is", Portfolio_Value)
        portfolio_Value_at_start=Portfolio_Value*(exp((-risk_free_rate)*time_to_expiry))
        print(portfolio_Value_at_start)
        option_price=- (portfolio_Value_at_start - delta*underlying_spot)
        return option_price
"""