# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:42:46 2018

@author: Feifan Xu, Kai Wang, Sina Malekian
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.core.pylabtools import figsize 
import scipy.stats as stats 
#import libraries we use in this project
pd.options.mode.chained_assignment = None

# UserWarning: JointGrid annotation is deprecated and will be removed in a future release
a=input('input the address of GDP data:')
# 'C:/Users/56284/OneDrive/study/quarter 4/data analysis ECE143/PROJECT/GDP.xlsx'
b=input('input the addess of Movies data:')
#'C:/Users/56284/OneDrive/study/quarter 4/data analysis ECE143/PROJECT/movies.csv'
#import data
def data_GDP(GDPDataDir=a):
    return pd.read_excel(GDPDataDir)
def data_Movies(MoviesDataDir=b):
    return pd.read_csv(MoviesDataDir)

#pick all movies and drop other kind of data
dff=data_Movies()
df = dff[dff['Type'] == 'movie']
df.index = range(len(df))

#process the data of US GDP
GDP = data_GDP()
GDP['GDP']=GDP['GDP'].astype(str)
move = dict.fromkeys((ord(c) for c in u"\xa0\n\t"))
GDP.GDP=GDP['GDP'].str.translate(move)
GDP['GDP']=pd.to_numeric(GDP['GDP'])
    
#change runtime to the format we want
#string to numbers
df['Runtime']=df['Runtime'].str.rstrip('nim ')
df.Runtime=pd.to_numeric(df.Runtime,errors='coerce') 

#change released time to the format we want
df['Released']=df['Released'].str.rstrip('0123456789')
df['Released']=df['Released'].str.rstrip('-')     #drop - of original data
df['releasedYear'], df['releasedMonth'] = df['Released'].str.split('-').str
df.releasedYear=pd.to_numeric(df.releasedYear,errors='coerce')
df.releasedMonth=pd.to_numeric(df.releasedMonth,errors='coerce') 

#create the type gbRatio
df['gbRatio']=df['Gross']/df['Budget']

#change the string of awards into wins and nominations
a = df.Awards.str.replace('.','&').str.split('&')    #drop punctuations in string
nomin = [0] * len(a)
win = [0] * len(a)
for i in range(len(a)):
    if(type(a[i]) == list):
        for y in a[i]:
            if('w' in y or 'W' in y):     #calc the number of wins
                for s in y.split():
                    if s.isdigit():
                        d = int(s)
                win[i] = win[i] + d
            if('nom' in y or 'Nom' in y):     #calc the number of nominations
                for s in y.split():
                    if s.isdigit():
                        d = int(s)
                nomin[i] = nomin[i] + d
                
#create the type wins and nominations
df['Wins']=win
df['Nomins']=nomin 

#Below we get some data we need per 5 years or 10 years, which is hard to process through numpy, so we use loop.
movies_earnmoney = []     #movies that earn money
movies_per5years = []
movies_highlyrated_imdb = []      #movies that over 7.4
averageGross_per5years = []     #gross of movies
averageLen_per5years = []     #length of title
averageLenGood_per5years = []
averageRT_per5years = []     #runtime
averageRT_highRate_per5years = []

#loop
for year in range(1885,2018,5):
    movies_earnmoney.append(len(df.loc[(df['Gross']/df['Budget'] > 1) & ((df['Year'] >= year)&(df['Year'] < year + 5))]))
    movies_per5years.append(len(df.loc[((df['Year'] >= year)&(df['Year'] < year + 5))]))
    movies_highlyrated_imdb.append(len(df.loc[(df['imdbRating'] > 7.4) & ((df['Year'] >= year)&(df['Year'] < year + 5))]))
    averageGross_per5years.append(df.loc[((df['Year'] >= year)&(df['Year'] < year + 5))].describe()['Gross']['mean'])
    averageLen_per5years.append(df.loc[((df['Year'] >= year)&(df['Year'] < year + 5))]['Title'].str.len().describe()['mean'])
    averageLenGood_per5years.append(df.loc[((df['imdbRating'] > 7.4)&(df['Year'] >= year)&(df['Year'] < year + 5))]['Title'].str.len().describe()['mean'])
    averageRT_per5years.append(df.loc[((df['Year'] >= year)&(df['Year'] < year + 5))].describe()['Runtime']['mean'])
    averageRT_highRate_per5years.append(df.loc[((df['Year'] >= year)&(df['Year'] < year + 5)&(df['imdbRating'] > 7.4))].describe()['Runtime']['mean'])

#the ratio of the number of movies that earn money over the total number of movies 
movies_moneyearn_ratio = [x/y for x,y in zip(movies_earnmoney,movies_per5years)]
movies_highlyrated_imdb_ratio = [x/y for x,y in zip(movies_highlyrated_imdb,movies_per5years)]
movies_total = len(df.loc[df['Gross']/df['Budget'] > 0]) 

#calculate the number of movies in each genre
Genre = {}
for G in df['Genre']:
    if not isinstance(G,str):
        continue
    GL = G.split(',')
    G_list = [[]* len(GL) for _ in range(len(GL))]
    for i in range(len(GL)):
        G_list[i] = GL[i].strip()
    for genre in G_list:
        if genre in Genre:
            Genre[genre] += 1
        else:
            Genre[genre] = 0

#set the default parameter of plots
figsize(20, 8)
font = {'weight': 'normal', 'size': 15}
sns.set(font_scale=1.5)

# Gigantic fonts
sns.set(font_scale=2)
plt.rc('font', **font)


#Below are plot funtions we will use in this project
def MHRY():
    '''
    Movies that earn money ,highly rated movies, ratio vs year grows
    '''
    y_earnratio = movies_moneyearn_ratio
    x_year = list(range(1885,2018,5))
    plt.figure(figsize=(20,8))
    plt.plot(x_year, y_earnratio,label = 'Movies Earn Money',linewidth=5.0,color='green')
    y_imdb_ratio = movies_highlyrated_imdb_ratio
    plt.plot(x_year, y_imdb_ratio,label = 'Highly Rated Movies',linewidth=5.0,color='orange')
    plt.legend()      #show the legend on plot
    plt.xlabel('Years')    #show the label on plot
    plt.ylabel('Ratio')
    plt.savefig('./movie_earnmoney and highlyrated_ratio.jpg',bbox_inches='tight')     #save the plot automatically
    plt.show()
    
def MGY():
    '''
    Movies gross and GDP vs years
    '''
    gross_per5years = []
    for year in range(1885,2018,5):
        gross_per5years.append((df.loc[((df['Year'] >= year)&(df['Year'] < year + 5))].describe()['Gross']['mean'])/10000000)
    y_gross = gross_per5years
    x_year = list(range(1885,2018,5))
    plt.figure(figsize=(20,8))
    plt.plot(x_year, y_gross,label = 'MoviesGross(Ten Million Dollars)',linewidth=5.0,color='green')
    y_GDP = GDP['GDP']
    x_GDP = GDP['Year\xa0']     #This can be a error in the OS system
    plt.plot(x_GDP, y_GDP,label = 'US GDP(Trillion Dollars)',linewidth=5.0,color='orange')
    plt.legend()
    plt.xlabel('Years')
    plt.savefig('./MoviesGross vs US GDP.jpg',bbox_inches='tight')
    plt.show()
    
def ALMY():
    '''
    Average length of movies vs year, in two categroies: all movies and high rated movies
    '''
    y_averLen = averageLen_per5years
    x_year = list(range(1885,2018,5))
    plt.figure(figsize=(20,8))
    plt.plot(x_year, y_averLen,label = 'All Movies',linewidth=5.0,color='green')
    y_averLenGood = averageLenGood_per5years
    plt.plot(x_year, y_averLenGood,label = 'Highly Rated Movies',linewidth=5.0,color='orange')
    plt.legend()
    plt.ylim((0, 40))    #limit the range of y showed on plot
    plt.xlabel('Years')
    plt.ylabel('Average Length of Movies Titles/Number of Letters')
    plt.savefig('./Rating vs length of name.jpg',bbox_inches='tight')
    plt.show()
    
def ARHY():
    '''
    Average runtime over years and high rated movies
    '''
    y_averRT = averageRT_per5years
    x_year = list(range(1885,2018,5))
    plt.figure(figsize=(20,8))
    plt.plot(x_year, y_averRT,label = 'All Movies',linewidth=5.0,color='green')
    plt.plot(x_year, averageRT_highRate_per5years,label = 'High Rated Movies',linewidth=5.0,color='orange')
    plt.legend()
    plt.ylim((0, 140))
    plt.xlabel('Years')
    plt.ylabel('Average Runtime of Movies(mins)')
    plt.savefig('./Runtime vs Years.jpg',bbox_inches='tight')
    plt.show()
    
def TLG1950():
    '''
    Title length vs gbRatio  1950 to 2018
    '''
    yearline=df[['Title','Year','gbRatio']]
    yearline['Title']=yearline['Title'].str.len()
    yearline=yearline[yearline['Year']>1950]
    yearline=yearline.groupby(pd.cut(yearline['Title'],np.arange(1,50,3)))
    titleYear=yearline.mean()[yearline.gbRatio.count()>30]
    titleYear=titleYear[['gbRatio']]
    plt.savefig("Title vs GBratio.jpg",bbox_inches='tight')
    titleYear.plot(kind = 'bar',rot = 0,figsize = (19,6))
    plt.xlabel('Length of Title(letters)')
    plt.ylabel('Mean of GrossBudget Ratio')
    
def RG():
    '''
    Runtime vs Gross
    '''
    rtgross=df[['Runtime','Gross']]
    rt=rtgross.groupby(pd.cut(df['Runtime'],np.arange(1,800,5)))
    rtdata=rt.mean()[rt.Gross.count()>20]
    plt.plot(rtdata['Runtime'], rtdata['Gross'], label = 'mean gross of movies',linewidth=5.0)
    plt.xlabel('Runtime(mins)')
    plt.ylabel('Mean gross of movies')
    
def NMG():
    '''
    number of movies by genre, each movie has several genres, 
    we choose to split them and choose the first one
    '''
    genresplit=df['Genre'].str.split(',',expand=True)
    genresplit['gbRatio']=df['gbRatio']
    a=genresplit[genresplit[0]!='Adult']
    a.index=range(len(a))
    genNum=a.groupby(0).count()
    genNum=genNum[genNum[1]>100]
    genNum=genNum[1]
    plt.savefig('./num by genre.png',bbox_inches='tight')
    genNum.plot(kind='barh',figsize=(15,10))
    plt.ylabel('Number of Movies')
    
def AGBRG():
    '''
    we plot the average gross budget ratio for every genre to 
    see which genres are more profitable
    '''
    genresplit=df['Genre'].str.split(',',expand=True)
    genresplit['gbRatio']=df['gbRatio']
    a=genresplit[genresplit[0]!='Adult']
    a.index=range(len(a))
    b=(a.groupby(0)['gbRatio'].mean().replace(0,None).dropna()) 
    plt.savefig('./gb ratio by genre bar.jpg',bbox_inches='tight')
    b.plot(kind='barh',figsize=(15,10))
    plt.axvline(1, color='k')
    plt.xlabel('Average GrossBudget Ratio')
    plt.ylabel('Genre')
    
def NGBRGC():
    '''Then, we want to see the number of movies that have gross budget ratio larger than one and less than one. 
    Of course we won't count the genres with very samll data
    '''
    genresplit=df['Genre'].str.split(',',expand=True)
    genresplit['gbRatio']=df['gbRatio']
    a=genresplit[genresplit[0]!='Adult']
    a.index=range(len(a))
    temp=a[a['gbRatio']>1]
    tempb=a[a['gbRatio']<=1]
    temp1=temp.groupby(0).count()
    temp2=tempb.groupby(0).count()
    temp1=temp1['gbRatio']
    temp2=-temp2['gbRatio']
    temp1=temp1[temp1>50]
    temp2=temp2[temp2<-40]
    gbRationLarge=temp1.rename('gbRatio>1')
    gbRatioSmall=temp2.rename('gbRatio<=1')
    gbRationLarge.plot(kind='bar',facecolor='#9999ff', edgecolor='white',figsize=(18,7),legend=True,rot=0)
    gbRatioSmall.plot(kind='bar',facecolor='#99ff99', edgecolor='white',figsize=(18,7),legend=True,rot=0)
    plt.ylabel('Number of Movies')
    plt.savefig('./bgRatio bar.png',bbox_inches='tight')
    
def NANTR():
    '''
    number of awards and nominations vs tomatoRatins to see 
    if there is some correlations
    '''
    winNomin=df[['Wins','Nomins','imdbRating','tomatoRating']]
    winNomin['Wins+Nomins']=winNomin['Wins']+winNomin['Nomins']
    winNomin=winNomin[['Wins+Nomins','imdbRating','tomatoRating']]
    winNomin=winNomin[winNomin['Wins+Nomins']>0]
    winNomin.index = range(len(winNomin))
    # Bigger than normal fonts
    sns.set(font_scale=1.5)
    # Gigantic fonts
    sns.set(font_scale=2)
    plt.rc('font', **font)
    f = sns.jointplot(x="tomatoRating", y="Wins+Nomins", data=winNomin,kind = 'reg',height=10,dropna = True,line_kws = {'color':'red'})
    f.annotate(stats.pearsonr)
    f.savefig("Reg tomato vs winNons.jpg",bbox_inches='tight')
    plt.ylim((-20, 300))
    
def NANIT():
    '''
    next is number of wins and nominations vs imdbRatings, 
    so we can have a comparison
    ''' 
    winNomin=df[['Wins','Nomins','imdbRating','tomatoRating']]
    winNomin['Wins+Nomins']=winNomin['Wins']+winNomin['Nomins']
    winNomin=winNomin[['Wins+Nomins','imdbRating','tomatoRating']]
    winNomin=winNomin[winNomin['Wins+Nomins']>0]
    winNomin.index = range(len(winNomin))
    # Bigger than normal fonts
    sns.set(font_scale=1.5)
    # Gigantic fonts
    sns.set(font_scale=2)
    plt.rc('font', **font)
    f = sns.jointplot(x="imdbRating", y="Wins+Nomins", data=winNomin,kind = 'reg', height=10,dropna = True,line_kws = {'color':'red'})
    f.annotate(stats.pearsonr)
    f.savefig("Reg imdb vs winNons.jpg")
    plt.ylim((-20, 300))
    
def GB():
    '''
    Then we want to see if there is some correlation between gross and budget
    '''
    dataBG=df[['Budget','Gross']]
    # Bigger than normal fonts
    sns.set(font_scale=1.5)
    # Gigantic fonts
    sns.set(font_scale=2)
    plt.rc('font', **font)
    f = sns.jointplot(x="Budget", y="Gross", data=dataBG,kind = 'reg', height=10,dropna = True,line_kws = {'color':'red'})
    f.annotate(stats.pearsonr)
    f.savefig("Reg Budget vs gross.jpg")
    
def GBRB():
    '''
    Next is find if there is correlation between gross budget ratio and budget
    '''
    dataBgb=df[['Budget','gbRatio']]
    # Bigger than normal fonts
    sns.set(font_scale=1.5)
    # Gigantic fonts
    sns.set(font_scale=2)
    plt.rc('font', **font)
    f = sns.jointplot(x="Budget", y='gbRatio', data=dataBgb,kind = 'reg', height=10,dropna = True,line_kws = {'color':'red'})
    f.annotate(stats.pearsonr)
    plt.ylim((0, 30))
    plt.xlim((0, 2e8))
    f.savefig("Reg Budget vs gbRatio.jpg")
    
def BR():
    '''
    Also we want to see if budget and ratings have correlations
    '''
    dataBtR=df[['Budget','tomatoRating']]
    # Bigger than normal fonts
    sns.set(font_scale=1.5)
    # Gigantic fonts
    sns.set(font_scale=2)
    plt.rc('font', **font)
    f = sns.jointplot(x="Budget", y="tomatoRating", data=dataBtR,kind = 'reg', height=10,dropna = True,line_kws = {'color':'red'})
    f.annotate(stats.pearsonr)
    f.savefig("Reg Budget vs rating.jpg",bbox_inches='tight')
    
def TRG():
    '''
    See if tomato ratings have correlations with gross
    '''
    dataTG=df[['tomatoRating','Gross']]
    # Bigger than normal fonts
    sns.set(font_scale=1.5)
    # Gigantic fonts
    sns.set(font_scale=2)
    plt.rc('font', **font)
    f = sns.jointplot(x="tomatoRating", y="Gross", data=dataTG,kind = 'reg', height=10,dropna = True,line_kws = {'color':'red'})
    f.annotate(stats.pearsonr)
    plt.ylim((0, 0.4e9))
    f.savefig("Reg Gross vs Rating.jpg",bbox_inches='tight')
    
def IRG():
    '''
    See if imdb ratings have correlations with gross
    Of coure we need IMDB ratings for comparison
    ''' 
    dataIG=df[['imdbRating','Gross']]
    # Bigger than normal fonts
    sns.set(font_scale=1.5)
    # Gigantic fonts
    sns.set(font_scale=2)
    plt.rc('font', **font) 
    f = sns.jointplot(x="imdbRating", y="Gross", data=dataIG,kind = 'reg', height=10,dropna = True,line_kws = {'color':'red'})
    f.annotate(stats.pearsonr)
    plt.ylim((0, 1e9))
    f.savefig("Reg imdbRating vs Gross.jpg",bbox_inches='tight')
    
def WNB():
    '''
    This is wins and nominations vs budget
    '''
    winNominB=df[['Wins','Nomins','Budget']]
    winNominB['Wins+Nomins']=winNominB['Wins']+winNominB['Nomins']
    winNominB=winNominB[['Wins+Nomins','Budget']]
    winNominB=winNominB[winNominB['Wins+Nomins']>0]
    winNominB.index = range(len(winNominB))
    # Bigger than normal fonts
    sns.set(font_scale=1.5)
    # Gigantic fonts
    sns.set(font_scale=2)
    plt.rc('font', **font)
    f = sns.jointplot(x="Budget", y="Wins+Nomins", data=winNominB,kind = 'reg', height=10,dropna = True,line_kws = {'color':'red'})
    f.annotate(stats.pearsonr)
    plt.ylim((-20, 300))
    f.savefig("Reg Budget vs winNons.jpg",bbox_inches='tight')
    
def RGBR1950():
    '''
    Runtime vs gbRatio  1950 2018
    '''
    timeGb=df[['Runtime','Year','Gross']]
    
    timeGb=timeGb[timeGb['Year']>1950]
    timeGb=timeGb.groupby(pd.cut(timeGb['Runtime'],np.arange(60,300,10)))
    runGb=timeGb.mean()[timeGb.Gross.count()>10]
    runGb=runGb[['Gross']]
    _,ax = plt.subplots()    #subplot
    runGb.plot(kind = 'bar',rot=0,figsize = (16,5),fontsize=12,legend=False, ax = ax)
    runGb.plot(kind = 'line',rot=0,figsize = (16,5),fontsize=12,legend=False, ax = ax,color = 'r',linewidth = 3.0)
    plt.ylabel('Gross',fontsize=16)
    plt.xlabel('Runtime',fontsize=16)
    plt.savefig('Runtime vs Gross.jpg',bbox_inches='tight')
    
def GDG():
    '''
    Gross vs domestic gross
    '''
    data_gross=df[['Domestic_Gross','Gross']]
    # Bigger than normal fonts
    sns.set(font_scale=1.5)
    # Gigantic fonts
    sns.set(font_scale=2)
    plt.rc('font', **font)
    f = sns.jointplot(x="Domestic_Gross", y="Gross", data=data_gross,kind = 'reg', height=10,dropna = True,line_kws = {'color':'red'})
    f.annotate(stats.pearsonr)
    plt.xlim((0, 4e8))
    plt.ylim((0, 1e9))
    f.savefig("Reg Domestic_Gross vs Gross.jpg")
    
def BRM():
    '''
    Box plot of released month
    '''
    df.boxplot(sym='r*', vert=False, column='Gross', by='releasedMonth', rot = 0,patch_artist=False,meanline=False,showmeans=True, notch = True,figsize=(20,8)) 
    plt.xlabel(u'Gross')
    plt.ylabel(u'releaseMonth')
    font = {'weight': 'normal', 'size': 15}
    plt.yticks([1.0, 2.0, 3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0], ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])     #change number into the name of months
    plt.rc('font', **font)
    plt.xlim((0, 6e8))
    plt.savefig('./boxplot release vs Gross.jpg',bbox_inches='tight')
    plt.show()
    
#End of the code