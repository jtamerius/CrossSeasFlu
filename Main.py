from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import stats
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib.patches as patches
import matplotlib.dates as mdates
from scipy import optimize
import pysal as ps
import linecache
import re
import xarray as xr
from scipy.interpolate import PchipInterpolator
import scipy.integrate as spi
import time


def CalcMaxRatio(qf):
    def MxRatio(df):
        df = df.sort_values('TestDate')
        ratio = df.iloc[1].Pos_A / df.iloc[0].Pos_A
        return ratio, df.iloc[1].Pos_A + df.iloc[0].Pos_A, df.iloc[0].Lat, df.iloc[0].Lon, df.iloc[0]['Zip Code']

    qf = qf.reset_index(drop=True)
    qf = qf.groupby('SofiaSN').filter(lambda g: len(g) == 2)
    qf_diff = qf.groupby('SofiaSN').apply(MxRatio)
    return qf_diff


def CalcMnthlyPercentPos(qf, flu_var):
    qf = AddMnth2df(qf)
    mnth_prc = qf.groupby('mnth').sum()
    return mnth_prc[flu_var] / mnth_prc['Total_Tests']


def Calc_Mnths_PercentPos(qf, flu_var, mnths):
    qf = AddMnth2df(qf)
    mnth_sum = qf.groupby('mnth').sum()
    mnth_prc = mnth_sum.loc[mnths].sum()

    return mnth_prc[flu_var] / mnth_prc['Total_Tests']


def CalcMonthlyProp(qf, flu_var):
    qf = AddMnth2df(qf)
    TT = qf[flu_var].sum()
    mnth_prop = qf.groupby('mnth').sum() / TT
    return mnth_prop[flu_var]


def PlotMonthlyProp(qf, grp, site):
    qf = AddMnth2df(qf)

    mnth_prop_qf = GetMonthlyProp(qf, 'Total_Tests')
    qf_site = FilterByGrp(qf, site, grp)
    mnth_prop_site = GetMonthlyProp(qf_site, 'Total_Tests')
    mnth_prop = mnth_prop_site - mnth_prop_qf
    ax = mnth_prop.plot(color='k')

    mnth_prop_qf = GetMonthlyProp(qf, 'Pos_A')
    qf_site = FilterByGrp(qf, site, grp)
    mnth_prop_site = GetMonthlyProp(qf_site, 'Pos_A')
    mnth_prop = mnth_prop_site - mnth_prop_qf
    ax = mnth_prop.plot(ax=ax, color='C1')

    mnth_prop_qf = GetMonthlyProp(qf, 'Pos_B')
    qf_site = FilterByGrp(qf, site, grp)
    mnth_prop_site = GetMonthlyProp(qf_site, 'Pos_B')
    mnth_prop = mnth_prop_site - mnth_prop_qf
    ax = mnth_prop.plot(ax=ax, color='b')

    ax.axhline(linestyle=':')
    plt.show()


def PlotMonthlyPerc(qf, grp, site):
    qf = AddMnth2df(qf)

    mnth_prop_all = CalcMnthlyPercentPos(qf, 'Pos_A')

    qf_site = FilterByGrp(qf, grp, site)
    mnth_prop_site = CalcMnthlyPercentPos(qf_site, 'Pos_A')
    mnth_prop_diff = mnth_prop_site - mnth_prop_all
    ax = mnth_prop_diff.plot(color='r')

    mnth_prop_all = CalcMnthlyPercentPos(qf, 'Pos_B')
    mnth_prop_site = CalcMnthlyPercentPos(qf_site, 'Pos_B')
    mnth_prop_diff = mnth_prop_site - mnth_prop_all
    ax = mnth_prop_diff.plot(color='b')

    ax.axhline(linestyle=':')
    plt.show()


def Calc_Prop4all(qf, grp, flu_var, mnths):
    qf_prop = qf.groupby(grp).apply(lambda g: CalcMonthlyProp(g, flu_var).loc[mnths].sum())
    qf_prop = pd.DataFrame(qf_prop, columns=['prop']).reset_index()
    qf = qf.groupby(grp).agg({flu_var: 'sum'}).reset_index()
    df = pd.merge(qf_prop, qf, on=grp)
    return df


def Calc_Clim4all(cf, grp, mnths):
    cf = cf[cf['mnth'].isin(mnths)]
    cf = cf.groupby(grp).agg({'temp': 'mean', 'sh': 'mean'}).reset_index()
    return cf


def Calc_Clim4cdc(cf, mnths):
    cf['mnth'] = cf['time'].dt.month
    cf = cf[cf['mnth'].isin(mnths)]
    cf = cf.groupby('City').agg({'temp': 'mean', 'sh': 'mean'}).reset_index()
    return cf


def run_regress4qf(flu_var, start_date, end_date, n_filter, flu_mnths, clim_mnths, grp):
    # load data
    qf = read_qf_data()

    # filter data
    qf = FilterByFirstTest(qf, start_date)
    qf = FilterByDate(qf, start_date, end_date)
    qf = FilterByNum(qf, flu_var, n_filter, grp)

    # group data
    qf_props = Calc_Prop4all(qf, grp, flu_var, flu_mnths)
    cf = read_cf_data(grp)
    cf_means = Calc_Clim4all(cf, grp, clim_mnths)
    df = pd.merge(qf_props, cf_means, on=grp)
    df = df.sort_values('sh')

    # Run regression model
    Y = pd.DataFrame(df.prop)
    df['sh2'] = df['sh'] ** 2
    # df['sh3'] = df['sh']**3

    X = df[['sh', 'sh2']]
    lm = linear_model.LinearRegression()
    model = lm.fit(X, Y, sample_weight=df[flu_var])
    y_pred = lm.predict(X)

    # plot data
    df.plot.scatter(x='sh', y='prop', s=df[flu_var] / 10)
    plt.plot(X['sh'], y_pred, color='blue', linewidth=1)
    plt.show()

    # Run statsmodel regression
    Y = pd.DataFrame(df.prop)
    W = df[flu_var]
    X = sm.add_constant(X)
    wls_model = sm.WLS(Y, X, weights=W)
    results = wls_model.fit()
    print(results.summary())

    return df


def run_regress4_hilo(flu_var, start_date, end_date, n_filter, grp, wnd_sz):
    # load data
    qf = read_qf_data()

    # filter data
    qf = FilterByFirstTest(qf, start_date)
    qf = FilterByDate(qf, start_date, end_date)
    qf = FilterByNum(qf, flu_var, n_filter, grp)
    qf = FilterByGaps(qf, n_filter, start_date, end_date)

    # group data
    qf = qf.groupby(grp).apply(rolling_hilo, flu_var=flu_var, start_date=start_date, end_date=end_date, wnd_sz=wnd_sz)
    qf = pd.DataFrame(qf)[0].apply(pd.Series)
    qf = qf.rename(columns={0: 'low', 1: 'hi', 2: 'low_mnth', 3: 'hi_mnth'})
    qf['ratio'] = qf['low'] / qf['hi']
    qf = qf.reset_index(drop=False)
    cf = read_cf_data(grp)
    cf_means = Calc_Clim4all(cf, grp, [11, 12, 1, 2])
    df = pd.merge(qf, cf_means, on=grp)

    # plot data
    df.plot.scatter(x='ratio', y='sh', s=df['hi'] / 4)
    plt.show()

    # Run regression model
    Y = pd.DataFrame(df.ratio)
    X = pd.DataFrame(df['sh'])
    lm = linear_model.LinearRegression()
    model = lm.fit(X, Y)

    print('R2 = ' + str(lm.score(X, Y)))

    slope, intercept, r_value, p_value, std_err = stats.linregress(X.values[:, 0], Y.values[:, 0])
    print('p-val: ' + str(p_value))
    print('slope: ' + str(slope))

    return df


def run_chi2(grp, wntr_mnths, flu_var, start_date, end_date):
    # n = 100
    # grp = 'metro'
    # wntr_mnths = [1, 2, 3, 10, 11, 12]
    # flu_var = 'Pos_AB'
    # start_date = '12-15-2016'
    # end_date = '12-15-2017'
    # run_chi2(grp, wntr_mnths, flu_var, start_date, end_date)

    def get_counts(qf, wntr_mnths, flu_var):
        col2 = qf[flu_var].sum()
        qf = qf[qf['mnth'].isin(wntr_mnths)]
        col1 = qf[flu_var].sum()
        vals1 = np.array([col1, col2])
        return vals1

    def run_test(vals1, vals2):
        # http://www.statisticshowto.com/benjamini-hochberg-procedure/
        obs = np.array([[vals1], [vals2]])
        g, p, dof, expctd = scs.chi2_contingency(obs)
        return p, vals2[0] / vals2[1]

    def do_it(grp, wntr_mnths, flu_var, vals1):
        vals2 = get_counts(grp, wntr_mnths, flu_var)
        [p, r] = run_test(vals1, vals2)
        return p, r

    def benjamini_hochberg(ps):
        df = pd.DataFrame(ps)[0].apply(pd.Series)
        df = df.rename(columns={0: 'p-val', 1: 'ratio'})
        df = df.sort_values(by='p-val')
        df = df.reset_index(drop=False)
        df['i'] = df['p-val'].rank()
        m = len(df)
        Q = 0.25
        df['q-val'] = df['i'] / m * Q
        df['sig'] = df['q-val'] > df['p-val']
        print(df)

    qf = read_qf_data()
    qf = FilterByFirstTest(qf, start_date)
    qf = FilterByDate(qf, start_date, end_date)
    qf = qf.groupby(grp).filter(lambda g: g[flu_var].sum() > 0)
    vals1 = get_counts(qf, wntr_mnths, flu_var)
    ps = qf.groupby(grp).apply(do_it, wntr_mnths=wntr_mnths, flu_var=flu_var, vals1=vals1)
    benjamini_hochberg(ps)


def get_flu_baseline_counts(g, flu_var, wnd_sz, start_date, end_date, hilo_epibase):
    g['TestDate'] = pd.to_datetime(g['TestDate'])
    g = FilterByDate(g, start_date, end_date)
    idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
    g = g.groupby('TestDate').sum()
    g.index = pd.DatetimeIndex(g.index)
    g = g.reindex(idx, fill_value=0)
    gg = g[flu_var].rolling(wnd_sz, min_periods=1, center=True).sum()
    mx = gg.max()

    if hilo_epibase == 'hilo':
        mn = gg.min()
    elif hilo_epibase == 'epibase':
        mn = g[flu_var].sum() - mx
        if mn == 0:
            mn = 1

    if pd.to_datetime(gg.idxmax() - round(wnd_sz / 2)) < pd.to_datetime(start_date):
        diff = pd.to_datetime(start_date) - pd.to_datetime(gg.idxmax() - round(wnd_sz / 2))
        strt_wnd = gg.idxmax() - round(wnd_sz / 2) + diff
        end_wnd = gg.idxmax() + round(wnd_sz / 2) + diff
    elif pd.to_datetime(gg.idxmax() + round(wnd_sz / 2)) > pd.to_datetime(end_date):
        diff = pd.to_datetime(end_date) - pd.to_datetime(gg.idxmax() - round(wnd_sz / 2))
        strt_wnd = gg.idxmax() - round(wnd_sz / 2) + diff
        end_wnd = gg.idxmax() + round(wnd_sz / 2) + diff
    else:
        strt_wnd = gg.idxmax() - round(wnd_sz / 2)
        end_wnd = gg.idxmax() + round(wnd_sz / 2)

    df = pd.DataFrame({'baseline': [int(mn)], 'epidemic': [int(mx)], 'strt_wnd': [strt_wnd], 'end_wnd': [end_wnd]})
    df['ratio'] = df.epidemic/df.baseline
    return df


def run_exact_test(df,sum_baseline, sum_epidemic):
    # http://www.statisticshowto.com/benjamini-hochberg-procedure/

    obs = np.array([[sum_baseline, sum_epidemic], [df.baseline, df.epidemic]])
    obs = obs.astype(int)
    obs[0][1] = obs[0][1] - obs[0][0]
    obs[1][1] = obs[1][1] - obs[1][0]
    oddr, p = scs.fisher_exact(obs)

    return p


def benjamini_hochberg(df):
    df = df.sort_values(by='p-val')
    df = df.reset_index(drop=False)
    df['i'] = df['p-val'].rank()
    m = len(df)
    Q = 0.01
    df['q-val'] = df['i'] / m * Q
    df['sig'] = ((df['q-val'] > df['p-val']) & (df['p-val'] <= 0.05))
    df['sig'] = df['sig'].astype(int)
    df[['strt_wnd', 'end_wnd']] = df[['strt_wnd', 'end_wnd']].astype('str')
    return df


def rolling_hilo(g, flu_var, wnd_sz, start_date, end_date, hilo_epibase):
    g['TestDate'] = pd.to_datetime(g['TestDate'])
    g = FilterByDate(g, start_date, end_date)
    idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
    g = g.groupby('TestDate').sum()
    g.index = pd.DatetimeIndex(g.index)
    g = g.reindex(idx, fill_value=0)
    gg = g[flu_var].rolling(wnd_sz, min_periods=1, center=True).sum()
    mx = gg.max()
    if hilo_epibase == 'hilo':
        mn = gg.min()
    elif hilo_epibase == 'epibase':
        mn = g[flu_var].sum() - mx
        if mn == 0:
            mn = 1

    if pd.to_datetime(gg.idxmax() - round(wnd_sz / 2)) < pd.to_datetime(start_date):
        diff = pd.to_datetime(start_date) - pd.to_datetime(gg.idxmax() - round(wnd_sz / 2))
        strt_wnd = gg.idxmax() - round(wnd_sz / 2) + diff
        end_wnd = gg.idxmax() + round(wnd_sz / 2) + diff
    elif pd.to_datetime(gg.idxmax() + round(wnd_sz / 2)) > pd.to_datetime(end_date):
        diff = pd.to_datetime(end_date) - pd.to_datetime(gg.idxmax() - round(wnd_sz / 2))
        strt_wnd = gg.idxmax() - round(wnd_sz / 2) + diff
        end_wnd = gg.idxmax() + round(wnd_sz / 2) + diff
    else:
        strt_wnd = gg.idxmax() - round(wnd_sz / 2)
        end_wnd = gg.idxmax() + round(wnd_sz / 2)

    return np.array([mn, mx]), strt_wnd, end_wnd


def MkMap(df, var, sz_var, lat_var, lon_var, f_out):
    # create new figure, axes instances.
    fig = plt.figure(figsize=(11.81, 9.09), dpi=600)
    ax = fig.add_axes([0.001, 0.001, .999, .999])

    # setup map projection:  c, l, i, h, f
    m = Basemap(llcrnrlon=-130., llcrnrlat=10, urcrnrlon=-50, urcrnrlat=46, \
                resolution='l', lon_0=-96, lat_0=40, projection='aea' \
                , ellps='GRS80', area_thresh=10000, lat_1=20, lat_2=60)

    ax.axis('off')
    fname_ST = 'Projects/QuidelData/QGIS/Qflu_QGIS_shpfiles/cb_2016_us_state_500k/cb_2016_us_state_500k'
    m.readshapefile(fname_ST, 'states', drawbounds=False)
    patches = []
    for info, shape in zip(m.states, m.states):
        patches.append(Polygon(np.array(shape), True))
    ax.add_collection(PatchCollection(patches, facecolor='#dedede', edgecolor='w', linewidths=.5, zorder=0))

    # define the color map
    cmap = pd.read_csv('Projects/QuidelData/Python/cmap.csv')
    cmap = cmap.as_matrix()
    cm = mplt.colors.ListedColormap(cmap / 255.0)

    Lon = df[lon_var].as_matrix()
    Lat = df[lat_var].as_matrix()
    var = df[var].as_matrix()
    sz_var = df[sz_var].as_matrix()

    points = m.scatter(Lon, Lat, latlon=True, s=sz_var / 10, cmap=cm, c=var, alpha=0.85, marker="o", linewidth=0)
    f = f_out + '.png'
    plt.savefig(f, dpi=600, bbox_inches='tight')


def nice_rolling_hilo_output(df):
    mn = df[0][0]
    mx = df[0][1]
    strt_wnd = df[1]
    end_wnd = df[2]
    strt_wnd = pd.to_datetime(strt_wnd)
    end_wnd = pd.to_datetime(end_wnd)
    return mn, mx, strt_wnd, end_wnd


def mk_df_prcs(grp, start_date, end_date, n_filter, flu_var, wnd_sz):
    def get_wndws_prc(g, flu_var, wnd_sz, start_date, end_date):
        g['TestDate'] = pd.to_datetime(g['TestDate'])
        idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
        g = g.groupby('TestDate').sum()
        g.index = pd.DatetimeIndex(g.index)
        g = g.reindex(idx, fill_value=0)
        gg = g[flu_var].rolling(wnd_sz, min_periods=1, center=True).sum()

        if pd.to_datetime(gg.idxmax() - round(wnd_sz / 2)) < pd.to_datetime(start_date):
            diff = pd.to_datetime(start_date) - pd.to_datetime(gg.idxmax() - round(wnd_sz / 2))
            strt_wnd = gg.idxmax() - round(wnd_sz / 2) + diff
            end_wnd = gg.idxmax() + round(wnd_sz / 2) + diff
        elif pd.to_datetime(gg.idxmax() + round(wnd_sz / 2)) > pd.to_datetime(end_date):
            diff = pd.to_datetime(end_date) - pd.to_datetime(gg.idxmax() - round(wnd_sz / 2))
            strt_wnd = gg.idxmax() - round(wnd_sz / 2) + diff
            end_wnd = gg.idxmax() + round(wnd_sz / 2) + diff
        else:
            strt_wnd = gg.idxmax() - round(wnd_sz / 2)
            end_wnd = gg.idxmax() + round(wnd_sz / 2)

        ins = (g.index > strt_wnd) & (g.index < end_wnd)
        prc_AB_epi = g[ins]['Pos_AB'].sum() / g[ins]['Total_Tests'].sum()
        prc_AB_bsl = g[~ins]['Pos_AB'].sum() / g[~ins]['Total_Tests'].sum()
        prc_A_epi = g[ins]['Pos_A'].sum() / g[ins]['Total_Tests'].sum()
        prc_A_bsl = g[~ins]['Pos_A'].sum() / g[~ins]['Total_Tests'].sum()
        prc_B_epi = g[ins]['Pos_B'].sum() / g[ins]['Total_Tests'].sum()
        prc_B_bsl = g[~ins]['Pos_B'].sum() / g[~ins]['Total_Tests'].sum()

        return strt_wnd, end_wnd, prc_AB_epi, prc_AB_bsl, prc_A_epi, prc_A_bsl, prc_B_epi, prc_B_bsl

    qf = Filter(grp, start_date, end_date, n_filter, flu_var)
    df = qf.groupby('mr').apply(get_wndws_prc, flu_var=flu_var, wnd_sz=wnd_sz, start_date=start_date, end_date=end_date)
    df = df.apply(pd.Series)
    df.columns = ['strt_wnd', 'end_wnd', 'prc_AB_epi', 'prc_AB_bsl', 'prc_A_epi', 'prc_A_bsl', 'prc_B_epi', 'prc_B_bsl']
    df = df.reset_index(drop=False)
    df[['strt_wnd', 'end_wnd']] = df[['strt_wnd', 'end_wnd']].astype('str')
    mk_shp(df, grp, 'EPA_SummerFlu/shp_out/Prc/prc')
    return df


def to_satscan(df):
    df = merge_lat_lon(df, 'metro')
    df[['epidemic', 'baseline']] = df[['epidemic', 'baseline']].astype('int')
    df[['epidemic']].to_csv('case_file.csv', index=True, index_label='ID')
    df[['baseline']].to_csv('baseline_file.csv', index=True, index_label='ID')
    # df = df.set_geometry('CityCenter')
    # df = df.to_crs({'init': 'epsg:31975'})
    # df['CityLon'] = df.CityCenter.apply(lambda p: p.x)
    # df['CityLat'] = df.CityCenter.apply(lambda p: p.y)
    df[['CityLat', 'CityLon']].to_csv('CoordinatesFile.csv', index=True, index_label='ID')


def pop_climate_not_working():
    df_pop = pd.read_pickle(
        'EPA_SummerFlu/PopulationData/PopData')
    df_mr = gpd.read_file('Projects/GIS_Data/MegaRegions_All/MR_All.shp')

    FullMx = gpd.sjoin(df_mr, df_pop, how='right')
    FullMx.dropna(inplace=True)
    FullMx = FullMx.rename(columns={'primeCity': 'mr'})
    FullMx = FullMx[FullMx['pop'] > 100000]
    FullMx = FullMx[FullMx['mr'] != 'AK']

    FullMx = FullMx.reset_index(drop=True)

    # import climate datasets
    t16 = xr.open_dataset(
        'EPA_SummerFlu/ClimateData/air.2m.gauss.2016.nc')

    # Temperature
    time16 = t16['time'].values
    T16 = pd.DataFrame(columns=['time', 'temp', 'mr'])

    def get_vals(df):
        print(df)
        vals = pd.DataFrame(t16.sel(lon=df.lons, lat=df.lats, method='nearest')['air'].values - 273.15 * df['pop'])
        vals.shape
        return vals

    def get_vals4mr(df):
        vals = df.apply(get_vals, axis=1)
        vals

    FullMx = FullMx.apply(get_vals, axis=1)
    FullMx




#####Plotting

def PlotTSgrp(grp, site, start_date, end_date, n_filter, flu_var):
    qf = Filter(grp, start_date, end_date, n_filter, flu_var)
    qf = FilterByGrp(qf, grp, site)
    ax = PlotTS(qf, flu_var)
    plt.show()
    return ax


def MapIt(qf, flu_var, fname):
    # create new figure, axes instances.
    fig = plt.figure(figsize=(11.81, 9.09), dpi=600)
    ax = fig.add_axes([0.001, 0.001, .999, .999])

    # setup map projection:  c, l, i, h, f
    m = Basemap(llcrnrlon=-130., llcrnrlat=10, urcrnrlon=-50, urcrnrlat=46, \
                resolution='l', lon_0=-96, lat_0=40, projection='aea' \
                , ellps='GRS80', area_thresh=10000, lat_1=20, lat_2=60)

    ax.axis('off')
    fname_ST = 'Projects/QuidelData/Python/QFluApp/Resources/Mapping/cb_2016_us_state_500k/cb_2016_us_state_500k'
    m.readshapefile(fname_ST, 'states', drawbounds=False)

    patches = []
    for info, shape in zip(m.states, m.states):
        patches.append(Polygon(np.array(shape), True))
    ax.add_collection(PatchCollection(patches, facecolor='#dedede', edgecolor='w', linewidths=.5, zorder=0))

    # read/make color map
    cmap = pd.read_csv('Projects/QuidelData/Python/cmap.csv')
    cmap = cmap.as_matrix()
    cm = mplt.colors.ListedColormap(cmap / 255.0)

    # start loop
    Lon = qf['lon'].as_matrix()
    Lat = qf['lat'].as_matrix()
    ratio = qf[flu_var].as_matrix()
    n = qf['n'].as_matrix()

    m.scatter(Lon, Lat, latlon=True, s=ratio * 5, cmap=cm, c=n, alpha=0.55, marker="o", linewidth=0)

    f = 'EPA_SummerFlu/' + fname + '.png'
    plt.savefig(f, dpi=600, bbox_inches='tight')


def PlotTS(qf, y):
    qf = qf.groupby('TestDate').apply(lambda g: g[y].sum())
    qf = qf.resample('D').pad()
    qf = qf.rolling(28, min_periods=1).mean()
    qf.plot(x='TestDate', y=y, linewidth=.5)


def PlotTSnorm(qf, grp, start_date,end_date, **kwargs):
    print(qf.mr.iloc[0])

    def norm(lat):
        if lat < 25:
            lat = 25.1
        elif lat > 45:
            lat = 44
        return (lat - 25) / float(44 - 25)

    if 'color' in kwargs:
        color = kwargs.get('color')
    else:
        #color = plt.cm.coolwarm(norm(qf.Lat.mean()))
        color = 'gray'

    if 'linewidth' in kwargs:
        linewidth = kwargs.get('linewidth')
    else:
        linewidth = .5

    if 'alpha' in kwargs:
        alpha = kwargs.get('alpha')
    else:
        alpha = 1



    qf = qf.groupby('TestDate').apply(lambda g: g[grp].sum())
    qf = qf.resample('D').pad()
    idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
    qf = qf.reindex(idx, fill_value=0)
    qf.index = pd.DatetimeIndex(qf.index)
    qf = qf.rolling(7, min_periods=1).mean()
    qf = qf / qf.sum()
    qf.plot(x='TestDate', y=grp, color=color, alpha=alpha, linewidth=linewidth)




#####Utility

def merge_lat_lon(df, grp):
    if grp == 'mr':
        shp = gpd.read_file('Projects/GIS_Data/MegaRegions_All/MR_All.shp')
        shp = shp.merge(df, left_on='primeCity', right_on='mr')

    elif grp == 'metro':
        shp = gpd.read_file(
            'Projects/GIS_Data/cb_2016_us_cbsa_20m/cb_2016_us_cbsa_20m.shp')
        # add centroids
        shp['CityCenter'] = shp.centroid
        shp = shp.merge(df, left_on='NAME', right_on='metro')
        shp['CityLon'] = shp.CityCenter.apply(lambda p: p.x)
        shp['CityLat'] = shp.CityCenter.apply(lambda p: p.y)

    return shp


def mk_shp(df, grp, file_out):
    if grp == 'mr':
        shp = gpd.read_file('Projects/GIS_Data/MegaRegions_All/MR_All.shp')
        shp.replace('HI', 'Honolulu', inplace=True)
        shp = shp.merge(df, left_on='primeCity', right_on='mr')
        shp = shp.set_geometry('geometry')
        shp['centroid'] = shp.centroid
        shp = shp.set_geometry('centroid')
        shp = shp.drop('geometry', 1)


    elif grp == 'metro':
        shp = gpd.read_file(
            'Projects/GIS_Data/cb_2016_us_cbsa_20m/cb_2016_us_cbsa_20m.shp')
        # add centroids
        shp['CityCenter'] = shp.centroid
        shp = shp.set_geometry('CityCenter')
        shp = shp.merge(df, left_on='NAME', right_on='metro')
        shp = shp.drop(['geometry'], 1)

    shp.to_file(file_out)
    return shp


def moving_average(df, start_date, end_date, flu_var, roll_wnd_sz):
    df['TestDate'] = pd.to_datetime(df['TestDate'])
    idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
    df = df.groupby('TestDate').sum()
    df.index = pd.DatetimeIndex(df.index)
    df = df.reindex(idx, fill_value=0)
    df = df[flu_var].rolling(roll_wnd_sz, min_periods=1, center=True).mean()
    return df


def AddMnth2df(qf):
    qf['mnth'] = qf['TestDate'].dt.month
    return qf



####Processsing and readign data

def process_data():
    def ImportQfluDataFiles():

        data_dir = 'Projects/QuidelData/Data/RawData'

        def search(keys, searchFor):
            for k in keys:
                if searchFor in keys[k]:
                    return k

        def Read_Qxls(fin):
            tmp = pd.read_excel(fin)
            Y = {'Facility Type': ['FacilityDescription', 'FacilityType', 'Facility Type'],
                 'Overall Result': ['OverallResult', 'Overall Result', '''Overall Result (DP's = Neg)'''],
                 'Zip Code': ['Zip Code', 'ZipCode', 'Zip'],
                 'SofiaSN': ['SofiaSerNum', 'Sofia Ser Num', 'SofiaSN'],
                 'Facility#': ['Facility#', 'Facility #', 'Facility Num', 'Facility'], 'City': ['City'],
                 'TestDate': ['TestDate', 'Date', 'Test Date'], 'FluA': ['Flu A', 'FluA', 'Result1'],
                 'FluB': ['Flu B', 'FluB', 'Result2'],
                 'County': ['County'], 'PatientAge': ['PatientAge', 'Age'], 'State': ['State', 'ST']}
            new_columns = []
            for hdr in list(tmp.columns.values):
                new_columns.append(search(Y, hdr))
            tmp.columns = new_columns
            return tmp

        FullMx = pd.DataFrame(
            columns=['SofiaSN', 'TestDate', 'Facility#', 'City', 'State', 'Zip Code', 'PatientAge', 'FluA', 'FluB',
                     'Overall Result', 'County', 'Facility Type'])

        for fin0 in glob.glob(data_dir + '/*.xlsx'):
            print(fin0)
            FullMx = pd.concat([Read_Qxls(fin0), FullMx])

        FullMx['Zip Code'] = FullMx['Zip Code'].astype('str').str[:5]

        return FullMx

    def GroupBySofiaTime(FullMx):
        def _ct_id_pos(grp):
            return grp['SofiaSN'].iloc[0], grp['Zip Code'].iloc[0], grp['State'].iloc[0], \
                   grp[grp.FluA == 'positive'].shape[
                       0], grp[grp.FluB == 'positive'].shape[0], grp.shape[0]

        FullMx_prime = FullMx.groupby(['TestDate', 'SofiaSN']).apply(_ct_id_pos).reset_index()
        FullMx_prime[['SofiaSN', 'Zip Code', 'ST', 'Pos_A', 'Pos_B', 'Total_Tests']] = FullMx_prime[0].apply(
            pd.Series)
        FullMx_prime.drop([0], axis=1, inplace=True)
        return FullMx_prime

    def GetLatLonFromZip(FullMx):
        print('Matching Lat/Lon with Zip Codes...')
        Zip2LatLon = pd.read_csv('Projects/QuidelData/Data/ZipCode2LatLon.csv')
        Zip2LatLon.columns = ['Zip Code', 'Lat', 'Lon']
        FullMx['Zip Code'] = FullMx['Zip Code'].astype(float).astype(int)
        FullMx = pd.merge(FullMx, Zip2LatLon, on='Zip Code', how='left')

        #############Clean up
        FullMx['TestDate'] = pd.to_datetime(FullMx.TestDate)
        FullMx.reset_index(inplace=True, drop=True)
        FullMx['Zip Code'] = FullMx['Zip Code'].astype('str').str[:5]
        FullMx['Zip Code'] = FullMx['Zip Code'].astype(float).astype(int)

        return FullMx

    def GetMrFromZip(FullMx):
        zips_mr = pd.read_pickle(
            'EPA_SummerFlu/Zip2Megaregions')
        FullMx = pd.merge(FullMx, zips_mr, on='Zip Code', how='left')
        FullMx = FullMx.drop(['plosName'], 1)
        FullMx = FullMx.rename(columns={'primeCity': 'mr'})
        FullMx['mr'][FullMx['ST'] == 'HI'] = 'HI'
        FullMx['mr'][FullMx['ST'] == 'PR'] = 'PR'
        FullMx['mr'][FullMx['ST'] == 'AK'] = 'AK'

        return FullMx

    def GetCityFromZip(FullMx):
        zips_city = pd.read_pickle('EPA_SummerFlu/Zip2City')
        FullMx = pd.merge(FullMx, zips_city, on='Zip Code', how='left')
        return FullMx

    FullMx = ImportQfluDataFiles()
    FullMx.to_pickle('Projects/Qflu2017_2018season/Qflu_Raw')
    print('Grouping by Sofia/time...')
    FullMx = GroupBySofiaTime(FullMx)
    print('Getting Lat/Lon from Zip...')
    FullMx = GetLatLonFromZip(FullMx)
    print('Getting Megaregion from Zip...')
    FullMx = GetMrFromZip(FullMx)
    print('Getting Metro from Zip...')
    FullMx = GetCityFromZip(FullMx)

    print('Saving...')
    FullMx.to_pickle('EPA_SummerFlu/Qflu_filtered')


def process_secondary():
    def GroupBySofiaTime(FullMx):
        def _ct_id_pos(grp):
            return grp['SofiaSN'].iloc[0], grp['Zip Code'].iloc[0], grp['State'].iloc[0], \
                   grp[grp.FluA == 'positive'].shape[
                       0], grp[grp.FluB == 'positive'].shape[0], grp.shape[0]

        FullMx_prime = FullMx.groupby(['TestDate', 'SofiaSN']).apply(_ct_id_pos).reset_index()
        FullMx_prime[['SofiaSN', 'Zip Code', 'ST', 'Pos_A', 'Pos_B', 'Total_Tests']] = FullMx_prime[0].apply(
            pd.Series)
        FullMx_prime.drop([0], axis=1, inplace=True)
        return FullMx_prime

    def GetLatLonFromZip(FullMx):
        print('Matching Lat/Lon with Zip Codes...')
        Zip2LatLon = pd.read_csv(
            'Projects/QuidelData/Data/RawData/ZipData/ZipCode2LatLon.csv')
        Zip2LatLon.columns = ['Zip Code', 'Lat', 'Lon']
        FullMx['Zip Code'] = FullMx['Zip Code'].astype(float).astype(int)
        FullMx = pd.merge(FullMx, Zip2LatLon, on='Zip Code', how='left')

        #############Clean up
        FullMx['TestDate'] = pd.to_datetime(FullMx.TestDate)
        FullMx.reset_index(inplace=True, drop=True)
        FullMx['Zip Code'] = FullMx['Zip Code'].astype('str').str[:5]
        FullMx['Zip Code'] = FullMx['Zip Code'].astype(float).astype(int)

        return FullMx

    def GetMrFromZip(FullMx):
        zips_mr = pd.read_pickle(
            'EPA_SummerFlu/Zip2Megaregions')
        FullMx = pd.merge(FullMx, zips_mr, on='Zip Code', how='left')
        FullMx = FullMx.drop(['plosName'], 1)
        FullMx = FullMx.rename(columns={'primeCity': 'mr'})
        FullMx['mr'][FullMx['ST'] == 'HI'] = 'HI'
        FullMx['mr'][FullMx['ST'] == 'PR'] = 'PR'
        FullMx['mr'][FullMx['ST'] == 'AK'] = 'AK'

        return FullMx

    def GetCityFromZip(FullMx):
        zips_city = pd.read_pickle('EPA_SummerFlu/Zip2City')
        FullMx = pd.merge(FullMx, zips_city, on='Zip Code', how='left')
        return FullMx

    FullMx = pd.read_pickle('Projects/Qflu2017_2018season/Qflu_Raw')
    print('Grouping by Sofia/time...')
    FullMx = GroupBySofiaTime(FullMx)
    print('Getting Lat/Lon from Zip...')
    FullMx = GetLatLonFromZip(FullMx)
    print('Getting Megaregion from Zip...')
    FullMx = GetMrFromZip(FullMx)
    print('Getting Metro from Zip...')
    FullMx = GetCityFromZip(FullMx)

    print('Saving...')
    FullMx.to_pickle('EPA_SummerFlu/Qflu_filtered')


def GetClimate4mr():
    #import city lat/lon data
    mr = pd.read_csv('EPA_SummerFlu/CityData/CityMR')
    mr['lon2'] = mr.longitude.mod(360)

    #import climate datasets
    t16 = xr.open_dataset('EPA_SummerFlu/ClimateData/air.2m.gauss.2016.nc')
    t17 = xr.open_dataset('EPA_SummerFlu/ClimateData/air.2m.gauss.2017.nc')
    t18 = xr.open_dataset('EPA_SummerFlu/ClimateData/air.2m.gauss.2018.nc')

    sh16 = xr.open_dataset('EPA_SummerFlu/ClimateData/shum.2m.gauss.2016.nc')
    sh17 = xr.open_dataset('EPA_SummerFlu/ClimateData/shum.2m.gauss.2017.nc')
    sh18 = xr.open_dataset('EPA_SummerFlu/ClimateData/shum.2m.gauss.2018.nc')

    #Temperature
    time16 = t16['time'].values
    time17 = t17['time'].values
    time18 = t18['time'].values
    T16 = pd.DataFrame(columns = ['time', 'temp', 'mr'])
    T17 = pd.DataFrame(columns = ['time', 'temp', 'mr'])
    T18 = pd.DataFrame(columns = ['time', 'temp', 'mr'])

    for i,row in mr.iterrows():
        temp16 = pd.DataFrame(t16.sel(lon=row.lon2, lat=row.latitude, method='nearest')['air'].values,columns=['temp'])
        temp16['mr'] = row.mr
        temp16['time']=time16
        T16 = pd.concat([T16,temp16])

        temp17 = pd.DataFrame(t17.sel(lon=row.lon2, lat=row.latitude, method='nearest')['air'].values,columns=['temp'])
        temp17['mr'] = row.mr
        temp17['time']= time17
        T17 = pd.concat([T17,temp17])

        temp18 = pd.DataFrame(t18.sel(lon=row.lon2, lat=row.latitude, method='nearest')['air'].values,columns=['temp'])
        temp18['mr'] = row.mr
        temp18['time']=time18
        T18 = pd.concat([T18,temp18])

    T = pd.concat([T16,T17,T18])
    T['temp']=T['temp']-273.15

    #Specific Humidity
    time16 = sh16['time'].values
    time17 = sh17['time'].values
    time18 = sh18['time'].values
    SH16 = pd.DataFrame(columns = ['time', 'sh', 'mr'])
    SH17 = pd.DataFrame(columns = ['time', 'sh', 'mr'])
    SH18 = pd.DataFrame(columns = ['time', 'sh', 'mr'])
    for i,row in mr.iterrows():
        s16 = pd.DataFrame(sh16.sel(lon=row.lon2, lat=row.latitude, method='nearest')['shum'].values,columns=['sh'])
        s16['mr'] = row.mr
        s16['time']=time16
        SH16 = pd.concat([SH16,s16])

        s17 = pd.DataFrame(sh17.sel(lon=row.lon2, lat=row.latitude, method='nearest')['shum'].values, columns=['sh'])
        s17['mr'] = row.mr
        s17['time'] = time17
        SH17 = pd.concat([SH17, s17])

        s18 = pd.DataFrame(sh18.sel(lon=row.lon2, lat=row.latitude, method='nearest')['shum'].values, columns=['sh'])
        s18['mr'] = row.mr
        s18['time'] = time18
        SH18 = pd.concat([SH18, s18])

    SH = pd.concat([SH16,SH17,SH18])

    Climate = pd.merge(T, SH, on=['mr', 'time'])

    Climate.to_pickle('EPA_SummerFlu/ClimateData/mrClimateData')


def GetClimate4metro():
    #import city lat/lon data
    mr = pd.read_csv('EPA_SummerFlu/CityData/MetroLatLon')
    mr['CityLon'] = mr['CityLon'].mod(360)

    #import climate datasets
    t16 = xr.open_dataset('EPA_SummerFlu/ClimateData/air.2m.gauss.2016.nc')
    t17 = xr.open_dataset('EPA_SummerFlu/ClimateData/air.2m.gauss.2017.nc')
    t18 = xr.open_dataset('EPA_SummerFlu/ClimateData/air.2m.gauss.2018.nc')

    sh16 = xr.open_dataset('EPA_SummerFlu/ClimateData/shum.2m.gauss.2016.nc')
    sh17 = xr.open_dataset('EPA_SummerFlu/ClimateData/shum.2m.gauss.2017.nc')
    sh18 = xr.open_dataset('EPA_SummerFlu/ClimateData/shum.2m.gauss.2018.nc')

    #Temperature
    time16 = t16['time'].values
    time17 = t17['time'].values
    time18 = t18['time'].values
    T16 = pd.DataFrame(columns = ['time', 'temp', 'metro'])
    T17 = pd.DataFrame(columns = ['time', 'temp', 'metro'])
    T18 = pd.DataFrame(columns = ['time', 'temp', 'metro'])
    for i,row in mr.iterrows():
        temp16 = pd.DataFrame(t16.sel(lon=row.CityLon, lat=row.CityLat, method='nearest')['air'].values,columns=['temp'])
        temp16['metro'] = row.NAME
        temp16['time']=time16
        T16 = pd.concat([T16,temp16])

        temp17 = pd.DataFrame(t17.sel(lon=row.CityLon, lat=row.CityLat, method='nearest')['air'].values,columns=['temp'])
        temp17['metro'] = row.NAME
        temp17['time']= time17
        T17 = pd.concat([T17,temp17])

        temp18 = pd.DataFrame(t18.sel(lon=row.CityLon, lat=row.CityLat, method='nearest')['air'].values,columns=['temp'])
        temp18['metro'] = row.NAME
        temp18['time']=time18
        T18 = pd.concat([T18,temp17])

    T = pd.concat([T16,T17,T18])
    T['temp']=T['temp']-273.15

    #Specific Humidity
    time16 = sh16['time'].values
    time17 = sh17['time'].values
    time18 = sh18['time'].values
    SH16 = pd.DataFrame(columns = ['time', 'sh', 'metro'])
    SH17 = pd.DataFrame(columns = ['time', 'sh', 'metro'])
    SH18 = pd.DataFrame(columns = ['time', 'sh', 'metro'])
    for i,row in mr.iterrows():
        s16 = pd.DataFrame(sh16.sel(lon=row.CityLon, lat=row.CityLat, method='nearest')['shum'].values,columns=['sh'])
        s16['metro'] = row.NAME
        s16['time']=time16
        SH16 = pd.concat([SH16,s16])

        s17 = pd.DataFrame(sh17.sel(lon=row.CityLon, lat=row.CityLat, method='nearest')['shum'].values, columns=['sh'])
        s17['metro'] = row.NAME
        s17['time'] = time17
        SH17 = pd.concat([SH17, s17])

        s18 = pd.DataFrame(sh18.sel(lon=row.CityLon, lat=row.CityLat, method='nearest')['shum'].values, columns=['sh'])
        s18['metro'] = row.NAME
        s18['time'] = time18
        SH18 = pd.concat([SH18, s18])

    SH = pd.concat([SH16,SH17,SH18])

    Climate = pd.merge(T, SH, on=['metro', 'time'])

    Climate.to_pickle('EPA_SummerFlu/ClimateData/metroClimateData')


def add_metro_areas():
    # Downlaoded data from here: https: // catalog.data.gov / dataset / tiger - line - shapefile - 2015 - nation - u - s - current - metropolitan - statistical - area - micropolitan - statist

    # load shape
    city = gpd.read_file(
        'Projects/GIS_Data/cb_2016_us_cbsa_20m/cb_2016_us_cbsa_20m.shp')
    # add centroids
    city['CityCenter'] = city.centroid

    # load zip code data
    Zip2LatLon = pd.read_csv(
        'Projects/QuidelData/Data/RawData/ZipData/ZipCode2LatLon.csv')
    Zip2LatLon.columns = ['Zip Code', 'Lat', 'Lon']
    geom = Zip2LatLon.apply(lambda x: Point([x['Lon'], x['Lat']]), axis=1)
    Zip2LatLon = gpd.GeoDataFrame(Zip2LatLon, geometry=geom)  # geom is a Series
    Zip2LatLon.crs = {'init': 'epsg:4326'}
    Zip2LatLon = Zip2LatLon.to_crs({'init': 'epsg:4269'})

    # join data
    zips_city = gpd.sjoin(Zip2LatLon, city, op='within', how='inner')

    # get lat/lon values
    zips_city['CityLon'] = zips_city.CityCenter.apply(lambda p: p.x)
    zips_city['CityLat'] = zips_city.CityCenter.apply(lambda p: p.y)

    # clean
    zips_city.drop(
        ['LSAD', 'ALAtf', 'AWATER', 'CSAFP', 'CBSAFP', 'AFFGEOID', 'index_right', 'GEOID', 'geometry', 'Lat',
         'Lon', 'CityCenter'], 1, inplace=True)

    # save
    zips_city.to_pickle('EPA_SummerFlu/Zip2City')


def read_qf_data():
    qf = pd.read_pickle('EPA_SummerFlu/Qflu_filtered')
    qf = qf.rename(columns={'NAME': 'metro'})
    qf['Pos_AB'] = qf['Pos_A'] + qf['Pos_B']
    qf = AddMnth2df(qf)
    qf = qf.replace('HI', 'Honolulu')
    return qf


def read_raw_qf_data():
    qf = pd.read_pickle('EPA_SummerFlu/Qflu_Raw')
    qf['TestDate'] = pd.to_datetime(qf['TestDate'])
    return qf


def read_cf_data(grp):
    if grp == 'mr':
        cf = pd.read_pickle(
            'EPA_SummerFlu/ClimateData/mrClimateData')
    elif grp == 'metro':
        cf = pd.read_pickle(
            'EPA_SummerFlu/ClimateData/metroClimateData')
        cf = cf.rename(columns={'mr': 'metro'})

    cf['time'] = pd.to_datetime(cf['time'])
    cf['mnth'] = cf.time.dt.month
    cf['sh'] = cf['sh'] * 1000
    return cf


def read_cdc_data():
    cdc = pd.read_pickle('EPA_SummerFlu/CDCflu/CDC122')
    cdc['Date'] = pd.to_datetime(cdc['Date'])
    cdc['mnth'] = cdc.Date.dt.month
    return cdc


def read_cf_cdc_data():
    cf_cdc = pd.read_pickle(
        'EPA_SummerFlu/ClimateData/ClimateData4cdc122')
    cf_cdc['time'] = pd.to_datetime(cf_cdc['time'])
    cf_cdc['mnth'] = cf_cdc.time.dt.month
    cf_cdc['sh'] = cf_cdc['sh'] * 1000
    return cf_cdc


def mk_gpd_df_pop():

    pop_counts = np.loadtxt('/Users/jamestamerius/Downloads/gpw-v4-population-count-rev10_2015_30_min_asc/gpw_v4_population_count_rev10_2015_30_min.asc', skiprows=6)

    ncols = linecache.getline("/Users/jamestamerius/Downloads/gpw-v4-population-count-rev10_2015_30_min_asc/gpw_v4_population_count_rev10_2015_30_min.asc", 1)
    ncols = int(re.search(r'\d+', ncols).group())

    nrows = linecache.getline("/Users/jamestamerius/Downloads/gpw-v4-population-count-rev10_2015_30_min_asc/gpw_v4_population_count_rev10_2015_30_min.asc", 2)
    nrows = int(re.search(r'\d+', nrows).group())

    xllcorner = linecache.getline("/Users/jamestamerius/Downloads/gpw-v4-population-count-rev10_2015_30_min_asc/gpw_v4_population_count_rev10_2015_30_min.asc", 3)
    xllcorner = int(re.search(r'\d+', xllcorner).group())*-1

    yllcorner = linecache.getline("/Users/jamestamerius/Downloads/gpw-v4-population-count-rev10_2015_30_min_asc/gpw_v4_population_count_rev10_2015_30_min.asc", 4)
    yllcorner = int(re.search(r'\d+', yllcorner).group())*-1

    cellsize = linecache.getline("/Users/jamestamerius/Downloads/gpw-v4-population-count-rev10_2015_30_min_asc/gpw_v4_population_count_rev10_2015_30_min.asc", 5)
    cellsize = float(re.findall("\d+\.\d+", cellsize)[0])

    lats = np.arange(yllcorner, yllcorner + nrows*cellsize, cellsize)
    lats = np.flipud(lats)
    lats = np.repeat(lats, 720, axis=0)
    lats = np.resize(lats, [290,720])
    lats = np.resize(lats,[208800,])

    lons = np.arange(xllcorner, xllcorner + ncols*cellsize, cellsize)
    lons = np.repeat(lons, 290, axis=0)
    lons = np.reshape(lons,[208800,1])
    lons = np.resize(lons, [720,290])
    lons = np.transpose(lons)
    lons = np.resize(lons,[208800,])

    pop_counts = np.resize(pop_counts,[208800,])

    df = pd.DataFrame({'lons':lons, 'lats':lats, 'pop': pop_counts})

    geom = df.apply(lambda x: Point([x['lons'], x['lats']]), axis=1)
    df = gpd.GeoDataFrame(df, geometry=geom)  # geom is a Series
    df.crs = {'init': 'epsg:4269'}

    df = df[df['pop'] > 0]

    df.to_pickle('EPA_SummerFlu/PopulationData/PopData')



#####Data Filters

def Filter(grp, start_date, end_date, n_filter, flu_var):
    qf = read_qf_data()
    qf = FilterByFirstTest(qf, start_date)
    qf = FilterByLastTest(qf, pd.to_datetime(end_date) - pd.to_timedelta(45, 'D'))
    qf = FilterByDate(qf, start_date, end_date)
    qf = FilterByNum(qf, flu_var, n_filter, grp)
    return qf


def FilterByNum(qf, flu_var, n, grp):
    qf = qf.groupby(grp).filter(lambda g: sum(g[flu_var]) > n)
    return (qf)


def FilterByFirstTest(qf, date):
    qf['TestDate'] = pd.to_datetime(qf['TestDate'])
    qf = qf.groupby(['SofiaSN', 'Zip Code']).filter(lambda g: g.TestDate.min() < pd.to_datetime(date))
    return (qf)


def FilterByLastTest(qf, date):
    qf['TestDate'] = pd.to_datetime(qf['TestDate'])
    qf = qf.groupby(['SofiaSN', 'Zip Code']).filter(lambda g: g.TestDate.max() > pd.to_datetime(date))
    return (qf)


def FilterByDate(qf, Date1, Date2):
    Date1 = pd.to_datetime(Date1)
    Date2 = pd.to_datetime(Date2)
    qf = qf[(qf['TestDate'] > Date1) & (qf['TestDate'] < Date2)]
    return qf


def FilterByGrp(qf, grp, site):
    qf = qf[qf[grp] == site]
    return qf


def FilterByMonth(qf, mnth):
    qf = qf[qf.TestDate.dt.month == mnth]
    return qf


def FilterByFreq(qf, n):
    # this filters out SF when the rate of testing gets too low
    def freq_func(grp, n):
        grp2 = grp.set_index('TestDate')
        grp2 = grp2.resample('W-MON').sum().fillna(0)
        zs = sum(grp2['Total_Tests'] == 0)
        if zs / len(grp) < n:
            return True
        else:
            return False

    qf = qf.groupby('SofiaSN').filter(lambda x: freq_func(x, n))
    return qf


def FilterByGaps(qf, wnd_sz, start_date, end_date):
    # this filters out SF when there are large gaps it testing
    # n is the size of gaps in days
    def gap_func(g, wnd_sz, start_date, end_date):
        g['TestDate'] = pd.to_datetime(g['TestDate'])
        g = g.sort_values('TestDate')
        idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
        g = g.set_index('TestDate')
        g = g.groupby('TestDate').agg({'Total_Tests': 'sum'})
        g = g.reindex(idx, fill_value=0)
        g = g['Total_Tests'].rolling(wnd_sz, min_periods=wnd_sz).sum()
        g = g.reindex(idx, fill_value=0)
        return sum(g == 0) == 0

    qf = qf.groupby(['SofiaSN', 'Zip Code']).filter(lambda x: gap_func(x, wnd_sz, start_date, end_date))
    return qf



####Make dataframe

def mk_final_df(grp, flu_var, start_date, end_date, wnd_sz, n_filter, hilo_epibase):

    def calc_Moran(df):
        df = df.to_crs({'proj': 'aea'})
        df = df.rename(columns={'centroid': 'geometry'})
        wt = ps.weights.DistanceBand.from_dataframe(df, threshold=600000, binary=True, silent=True)
        wt.transform = 'r'
        mi = ps.Moran(np.log(df['ratio']), wt, two_tailed=False, permutations=1000)
        print(wt.histogram)
        print('Moran''s I: ', str(mi.I))
        print('Moran''s I p-val:', str(mi.p_sim))


    qf = Filter(grp, start_date, end_date, n_filter, flu_var)
    df = qf.groupby(grp).apply(get_flu_baseline_counts, flu_var=flu_var, wnd_sz=wnd_sz, start_date=start_date, end_date=end_date, hilo_epibase=hilo_epibase)
    df = df.reset_index()
    df = df.drop('level_1',1)
    p_vals = df.groupby(grp).apply(run_exact_test,sum_baseline=df['baseline'].sum(), sum_epidemic=df['epidemic'].sum())
    df = pd.merge(df, p_vals.reset_index(), on=grp)
    df = df.rename(columns={0:'p-val'})
    df = benjamini_hochberg(df)
    df['ratio_diff'] = df['ratio'] / df['ratio'].median()
    df['log_ratio'] = df['ratio'].apply(np.log10)

    print('median ratio: ' + str(df['ratio'].median()))
    print('mean ratio: ' + str(df['ratio'].mean()))
    print('pooled-mean ratio: ' + str(df.epidemic.sum() / df.baseline.sum()))


    file_out = 'shp_out/' + str(pd.to_datetime(start_date).year) + '_' + str(pd.to_datetime(start_date).year + 1) + '_' + flu_var
    df = mk_shp(df, grp, file_out)

    df['total'] = df['epidemic']+df['baseline']
    df = df.drop(['plosName', 'regionName', 'primeCity', 'index'], 1)

    return df


def mk_final_df_clim(grp, flu_var, start_date, end_date, wnd_sz, n_filter, hilo_epibase):

    def get_clim_vals(df, cf, qf, grp):
        cf = cf[cf[grp] == df[grp].iloc[0]]
        qf = qf[qf[grp] == df[grp].iloc[0]]
        epi_time = (cf['time'] > pd.to_datetime(df['strt_wnd'].iloc[0])) & (
        cf['time'] < pd.to_datetime(df['end_wnd'].iloc[0]))
        bsl_time = ~epi_time
        cf_epi = cf[epi_time]
        cf_bsl = cf[bsl_time]

        qcf_epi = pd.merge(qf, cf_epi, left_on='TestDate', right_on='time')
        qcf_bsl = pd.merge(qf, cf_bsl, left_on='TestDate', right_on='time')

        var_epi = 'sh_flu'
        var_bsl = 'SH_bsl'
        var_epi2 = 'sh_flu2'
        var_bsl2 = 'SH_bsl2'
        var_min = 'SH_min'
        var_max = 'SH_max'

        t_epi = 'tmp_flu'
        t_bsl = 'tmp_bsl'
        t_epi2 = 'tmp_flu2'
        t_bsl2 = 'tmp_bsl2'
        t_min = 'tmp_min'
        t_max = 'tmp_max'

        df[var_epi] = (qcf_epi[flu_var] * qcf_epi['sh']).sum() / (qcf_epi[flu_var].sum())
        df[var_min] = (qcf_epi['sh'].min())
        df[var_max] = (qcf_epi['sh'].max())

        if qcf_bsl[flu_var].sum() != 0:
            df[var_bsl] = (qcf_bsl[flu_var] * qcf_bsl['sh']).sum() / (qcf_bsl[flu_var].sum())
        else:
            df[var_bsl] = (qcf_bsl['sh']).median()

        df[var_bsl2] = df[var_bsl] ** 2
        df[var_epi2] = df[var_epi] ** 2



        df[t_epi] = (qcf_epi[flu_var] * qcf_epi['temp']).sum() / (qcf_epi[flu_var].sum())
        df[t_min] = (qcf_epi['temp'].min())
        df[t_max] = (qcf_epi['temp'].max())

        if qcf_bsl[flu_var].sum() != 0:
            df[t_bsl] = (qcf_bsl[flu_var] * qcf_bsl['temp']).sum() / (qcf_bsl[flu_var].sum())
        else:
            df[t_bsl] = (qcf_bsl['temp']).median()

        df[t_bsl2] = df[t_bsl] ** 2
        df[t_epi2] = df[t_epi] ** 2

        return df


    cf = read_cf_data(grp)
    cf = cf.replace('HI', 'Honolulu')
    df = mk_final_df(grp, flu_var, start_date, end_date, wnd_sz, n_filter, hilo_epibase)
    qf = Filter(grp, start_date, end_date, n_filter, flu_var)
    df = df.groupby(grp).apply(get_clim_vals, cf=cf, qf=qf, grp=grp)
    df = df.drop(['i'],1)

    return df


def full_df_to_stata():

    def strip_df(df):
        df['lat'] = df.geometry.y
        df['lon'] = df.geometry.x

        df = df.drop(['baseline', 'end_wnd', 'epidemic', 'strt_wnd', 'ratio', 'p-val', 'q-val', 'sig', 'ratio_diff',  'centroid'],1)

        return df

    def add_details_df(df, flu_var,start_date):
        year = start_date[:4]
        df['type'] = flu_var
        df['year'] = year
        df[str(flu_var)+str(year)] = 1



        return df

    df = pd.DataFrame()

    flu_var = ['Pos_A', 'Pos_B']
    start_date = ['2016-7-1', '2017-7-1']
    end_date = ['2017-6-30', '2018-5-10']

    wnd_sz = 270
    n_filter = 150
    grp = 'mr'
    hilo_epibase = 'epibase'

    for t in zip(start_date, end_date):

        for var in flu_var:
            tmp_df = mk_final_df_clim(grp, var, t[0], t[1], wnd_sz, n_filter, hilo_epibase)
            tmp_df = strip_df(tmp_df)
            tmp_df = add_details_df(tmp_df, var, t[0])
            df = pd.concat([df,tmp_df])

    df = df.fillna(0)
    df['dummy'] = df['Pos_A2016']+df['Pos_B2016']*2+df['Pos_A2017']*3+df['Pos_B2017']*4

    popvac = pd.read_excel(
        'EPA_SummerFlu/AuxData/RegionPopAndVaccRates.xlsx')
    popvac = popvac.replace('Hawaii', 'Honolulu')
    df = pd.merge(df, popvac, left_on='mr', right_on='Region')
    df = df.drop('Region', 1)

    df.to_csv('EPA_SummerFlu/HierarchicalModel/df_out.csv')


    return df



####Figures

def mk_all_figs(grp, start_date1, end_date1, start_date2, end_date2, wnd_sz, n_filter, hilo_epibase):
    mk_fig1(start_date1, end_date1, start_date2, end_date2, grp, n_filter, wnd_sz)
    mk_fig3(grp, start_date1, end_date1, start_date2, end_date2, wnd_sz, n_filter, hilo_epibase)
    mk_fig4(grp, start_date1, end_date1 , start_date2, end_date2, wnd_sz, n_filter, hilo_epibase)
    #mk_fig5(grp, start_date1, end_date1, start_date2, end_date2, wnd_sz, n_filter, hilo_epibase)


def mk_fig1(start_date1, end_date1, start_date2, end_date2, grp, n_filter, wnd_sz):


    def mk_ts_plots4fig1(start_date, end_date, flu_var, grp, n_filter, wnd_sz):
        hilo_epibase = 'epibase'

        file_out = str(start_date) + '_' + flu_var
        f_out = 'EPA_SummerFlu/Figures/Fig1/' + file_out + '.png'

        qf = Filter(grp, start_date, end_date, n_filter, flu_var)

        df = qf.groupby(grp).apply(rolling_hilo, flu_var=flu_var, wnd_sz=wnd_sz, start_date=start_date,
                                   end_date=end_date, hilo_epibase=hilo_epibase)
        df = pd.DataFrame(df)[0].apply(pd.Series)
        df = df.rename(columns={0: 'counts', 1: 'strt_wnd', 2: 'end_wnd'})
        df['strt_wnd'] = pd.to_datetime(df['strt_wnd'])
        df['end_wnd'] = pd.to_datetime(df['end_wnd'])
        ax = plt.axes()
        qf.groupby(grp).apply(PlotTSnorm,start_date=start_date,end_date=end_date, grp=flu_var, alpha=.5,linewidth=.25)

        PlotTSnorm(qf[qf.mr=='Orlando'],flu_var,start_date,end_date, alpha=.5,linewidth=2,color='C3')
        PlotTSnorm(qf[qf.mr=='Des Moines'],flu_var,start_date,end_date, alpha=.5,linewidth=2,color='C0')


        ylims = ax.get_ylim()
        strt = df.set_index('strt_wnd')
        strt = strt['counts'].apply(pd.Series)
        strt['ymax'] = ylims[0] - ((strt[0] + strt[1]) / strt[1].max() * ylims[0])

        end = df.set_index('end_wnd')
        end = end['counts'].apply(pd.Series)
        end['ymax'] = ylims[0] - ((end[0] + end[1]) / end[1].max() * ylims[0])

        y = [-0.00125] * len(strt.index)
        #ax.vlines(x=strt.index, ymax=-0.0001, ymin=-0.005,alpha=0.04, linewidth=1, color='r')
        # ax.vlines(x=end.index, ymax=-0.0001, ymin=-0.005,alpha=0.25, linewidth=1, color='k')

        ax.hlines(y=y,xmin=strt.index, xmax=end.index, alpha=0.03, linewidth=5, color='k')
        #ax.hlines(x=end.index, ymax=-0.0001, ymin=-0.005, alpha=0.25, linewidth=1, color='k')

        ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
        ax.set_ylim([-0.0025, 0.03])
        plt.minorticks_off()
        plt.xlabel(' ')
        plt.ylabel('Case Proportion')

        plt.tight_layout()
        fig = ax.get_figure()
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        fig.savefig(f_out, dpi=300)
        plt.close(fig)

    flu_var = ['Pos_A', 'Pos_B']

    for var in flu_var:
        mk_ts_plots4fig1(start_date1, end_date1, var, grp, n_filter, wnd_sz)
        mk_ts_plots4fig1(start_date2, end_date2, var, grp, n_filter, wnd_sz)


def mk_fig2(grp, flu_var, start_date, end_date, wnd_sz, n_filter):
    file_out = str(pd.to_datetime(start_date).year) + '_' + str(pd.to_datetime(start_date).year + 1) + '_' + flu_var
    f_out = 'EPA_SummerFlu/shp_out/' + file_out
    df = mk_final_df(grp, flu_var, start_date, end_date, wnd_sz, n_filter, 'epibase')
    mk_shp(df, grp, f_out)


def mk_fig2_180(grp, flu_var, start_date, end_date, wnd_sz, n_filter):
    file_out = str(pd.to_datetime(start_date).year) + '_' + str(pd.to_datetime(start_date).year + 1) + '_' + flu_var
    f_out = 'EPA_SummerFlu/shp_out_180/' + file_out
    df = mk_final_df(grp, flu_var, start_date, end_date, wnd_sz, n_filter, 'epibase')
    mk_shp(df, grp, f_out)


def mk_fig2_330(grp, flu_var, start_date, end_date, wnd_sz, n_filter):
    file_out = str(pd.to_datetime(start_date).year) + '_' + str(pd.to_datetime(start_date).year + 1) + '_' + flu_var
    f_out = 'EPA_SummerFlu/shp_out_330/' + file_out
    df = mk_final_df(grp, flu_var, start_date, end_date, wnd_sz, n_filter, 'epibase')
    mk_shp(df, grp, f_out)


def mk_fig3(grp, start_date1, end_date1 , start_date2, end_date2, wnd_sz, n_filter, hilo_epibase):

    def brkpt_reg(df):

        x = df['lat'].values
        y = df['ratio'].values

        def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x < x0, x >= x0],
                                [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])

        p, e = optimize.curve_fit(piecewise_linear, x, y)
        xd = df['lat'].values
        plt.plot(xd, piecewise_linear(xd, *p),color='k',linewidth=.8)

    def Fig_RatioByLat(grp, flu_var, start_date, end_date, wnd_sz, n_filter, hilo_epibase):

        def btstrp(df):

            # configure bootstrap
            n_iterations = 1000

            tot_pos = df['epidemic'] + df['baseline']
            values = np.zeros(int(tot_pos))
            values[0:int(df['baseline'])] = 1

            # run bootstrap
            test = np.empty([0, 1])
            for i in range(n_iterations):
                tmp = np.random.choice(values, int(tot_pos))
                tmp = len(tmp) / np.sum(tmp)
                test = np.append(test, tmp)

            # confidence intervals
            df['upperCI'] = np.percentile(test, 97.5)
            df['lowerCI'] = np.percentile(test, 2.5)

            return df

        file_out = flu_var + start_date[:4]
        f_out = 'EPA_SummerFlu/Figures/RatioByLat/' + file_out + '.png'

        df = mk_final_df(grp, flu_var, start_date, end_date, wnd_sz, n_filter, hilo_epibase)
        df['lat'] = df.geometry.y
        tmp = np.empty([0, 4])
        for i in np.arange(25, 55, .5):
            ins = (df['lat'] > i - 2) & (df['lat'] < i + 2)
            tmp = np.append(tmp, np.array([[i, ins.sum(), df[ins]['baseline'].sum(), df[ins]['epidemic'].sum()]]),
                            axis=0)

        df2 = pd.DataFrame(tmp, columns=['lat', 'n', 'baseline', 'epidemic'])
        df2['ratio'] = df2['epidemic'] / df2['baseline']
        df2 = df2.dropna()
        df3 = df2.apply(btstrp, axis=1)
        df3['upperCI'].replace(np.inf, df3['upperCI'][df3['upperCI'] != np.inf].max(), inplace=True)

        if pd.to_datetime(df['strt_wnd'].min()) < pd.to_datetime('2017-6-30'):
            color = 'C0'
        else:
            color = 'C1'

        ax = plt.axes()

        # df5 = df.apply(btstrp, axis=1)
        # df5['upperCI'].replace(np.inf, 5000, inplace=True)
        # ax.vlines(x=df5.lat, ymax=df5.upperCI, ymin=df5.lowerCI, linewidth=.5, color='gray', alpha=.2)

        df4 = df[['lat', 'ratio']]
        df4.plot.scatter(x='lat', y='ratio', color='white', edgecolor='gray',alpha=.2,
                         s=(df2['baseline'] + df2['epidemic']) / 100, ax=ax)

        plt.fill_between(df3['lat'], df3['upperCI'], df3['lowerCI'], color=color, alpha=.15, linewidth=0.0)


        df3.plot(x='lat', y='ratio', legend=False, color=color, ax=ax)


        brkpt_reg(df2)


        ax.set_ylim([0, df3['ratio'].max() * 2])
        plt.axhline(y=df2['ratio'].median(), linestyle='--', color='k', linewidth=.25)
        plt.ylabel('Cross-Seasonal Ratio')
        plt.xlabel('Latitude')
        fig = ax.get_figure()
        fig.savefig(f_out)
        plt.clf()

    flu_var = ['Pos_A', 'Pos_B']

    for var in flu_var:
        Fig_RatioByLat(grp, var, start_date1, end_date1, wnd_sz, n_filter, hilo_epibase)
        Fig_RatioByLat(grp, var, start_date2, end_date2, wnd_sz, n_filter, hilo_epibase)


def mk_fig4(grp, start_date1, end_date1 , start_date2, end_date2, wnd_sz, n_filter, hilo_epibase):

    def mk_fig_ts_plots(start_date1, end_date1, start_date2, end_date2, flu_var, grp, site, n_filter, wnd_sz,
                        hilo_epibase):
        file_out = site + '_' + flu_var
        f_out = 'EPA_SummerFlu/Figures/Fig4/' + file_out + '.png'

        e1 = pd.to_datetime(end_date1) + pd.to_timedelta(30, 'D')
        qf1 = Filter(grp, start_date1, e1, n_filter, flu_var)
        qf1 = FilterByGrp(qf1, grp, site)
        df1 = rolling_hilo(qf1, flu_var, wnd_sz, start_date1, end_date1, hilo_epibase)
        [mn1, mx1, strt_wnd1, end_wnd1] = nice_rolling_hilo_output(df1)

        s2 = pd.to_datetime(start_date1) - pd.to_timedelta(30, 'D')
        qf2 = Filter(grp, s2, end_date2, n_filter, flu_var)
        qf2 = FilterByGrp(qf2, grp, site)
        df2 = rolling_hilo(qf2, flu_var, wnd_sz, start_date2, end_date2, hilo_epibase)
        [mn2, mx2, strt_wnd2, end_wnd2] = nice_rolling_hilo_output(df2)

        qf1 = moving_average(qf1, start_date1, end_date1, flu_var, 7)
        qf1 = qf1 / qf1.sum()
        ax = qf1.plot(y=flu_var, title=site, legend=False, color='k',alpha=.8)
        plt.ylabel(flu_var)

        qf2 = moving_average(qf2, start_date2, end_date2, flu_var, 7)
        qf2 = qf2 / qf2.sum()
        qf2.plot(y=flu_var, ax=ax, legend=False, color='k',alpha=.8)

        start1 = mdates.date2num(strt_wnd1)
        end1 = mdates.date2num(end_wnd1)
        width1 = end1 - start1

        start2 = mdates.date2num(strt_wnd2)
        end2 = mdates.date2num(end_wnd2)
        width2 = end2 - start2

        ylims = ax.get_ylim()

        rect1 = patches.Rectangle((strt_wnd1, -1), width1, ylims[1] + 10, linewidth=.25, edgecolor='k', facecolor='k',
                                  alpha=.05)
        ax.add_patch(rect1)

        rect2 = patches.Rectangle((strt_wnd2, -1), width2, ylims[1] + 10, linewidth=.25, edgecolor='k', facecolor='k',
                                  alpha=.05)
        ax.add_patch(rect2)
        fig = ax.get_figure()
        plt.ylabel(' ')
        ax.hlines(y=0, xmin=start_date1, xmax=end_date2, linewidth=1, color='k')
        ax.set_ylim([-.001, 0.03])
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        fig.savefig(f_out)
        plt.clf()

    flu_var = ['Pos_A', 'Pos_B']
    sites = ['Minneapolis','Des Moines','Washington DC', 'Houston', 'Birmingham', 'Orlando', 'Honolulu']

    for var in flu_var:
        for site in sites:
            mk_fig_ts_plots(start_date1, end_date1, start_date2, end_date2, var, grp, site, n_filter, wnd_sz,
                        hilo_epibase)


def mk_fig5(grp, start_date1, end_date1, start_date2, end_date2, wnd_sz, n_filter, hilo_epibase):


    flu_var = ['Pos_A', 'Pos_B']

    for var in flu_var:
        run_spatial_reg(grp, var, start_date1, end_date1, wnd_sz, n_filter, hilo_epibase)
        run_spatial_reg(grp, var, start_date2, end_date2, wnd_sz, n_filter, hilo_epibase)


def mk_Supp_Maps(grp, start_date1, end_date1, start_date2, end_date2, n_filter):
    flu_var = ['Pos_A', 'Pos_B']

    for var in flu_var:
        mk_fig2_180(grp, var, start_date1, end_date1, 180, n_filter)
        mk_fig2_180(grp, var, start_date2, end_date2, 180, n_filter)
        mk_fig2_330(grp, var, start_date1, end_date1, 330, n_filter)
        mk_fig2_330(grp, var, start_date2, end_date2, 330, n_filter)



####Analysis

def get_avg_start_end_dates(grp, flu_var, start_date, end_date, wnd_sz, n_filter):

    df = run_fishexact_hilo(grp, flu_var, start_date, end_date, wnd_sz, n_filter, 'epibase')
    df['doy_strt'] = pd.to_datetime(df['strt_wnd']).dt.dayofyear
    df['doy_end'] = pd.to_datetime(df['end_wnd']).dt.dayofyear
    print('flu_var = ' + flu_var)
    print('start_date = ' + start_date)
    print('end_date = ' + end_date)

    print('start_wndw = ' + df[df.doy_strt == df.doy_strt.mean().round()].iloc[0].strt_wnd)
    print('end_wndw = ' + df[df.doy_end == df.doy_end.mean().round()].iloc[0].end_wnd)
    print('Num of subregions: ' + str(df[grp].unique().size))


def mk_supp_fig1(grp, start_date1, end_date1, start_date2, end_date2, n_filter, wnd_sz):


    def mk_df_prcs(grp, start_date, end_date, n_filter, flu_var, wnd_sz):


        def brkpt_reg(df):

            x = df['lat'].values
            y = df['ratio'].values

            def piecewise_linear(x, x0, y0, k1, k2):
                return np.piecewise(x, [x < x0, x >= x0],
                                    [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])

            p, e = optimize.curve_fit(piecewise_linear, x, y)
            xd = df['lat'].values
            plt.plot(xd, piecewise_linear(xd, *p),color='k',linewidth=.8)


        def btstrp(df):
            # configure bootstrap
            n_iterations = 1000

            values_bsl = np.zeros(int(df['Total_Tests_bsl']))
            values_bsl[0:int(df['TotPos_bsl'])] = 1

            values_epi = np.zeros(int(df['Total_Tests_epi']))
            values_epi[0:int(df['TotPos_epi'])] = 1

            # run bootstrap
            test_bsl = np.empty([0, 1])
            test_epi = np.empty([0, 1])

            for i in range(n_iterations):
                tmp_bsl = np.random.choice(values_bsl, len(values_bsl))
                tmp_bsl = np.sum(tmp_bsl)/len(tmp_bsl) * 100
                test_bsl = np.append(test_bsl, tmp_bsl)

                tmp_epi = np.random.choice(values_epi, len(values_epi))
                tmp_epi = np.sum(tmp_epi) / len(tmp_epi) * 100
                test_epi = np.append(test_epi, tmp_epi)

            # confidence intervals
            df['upperCI_bsl'] = np.percentile(test_bsl, 97.5)
            df['lowerCI_bsl'] = np.percentile(test_bsl, 2.5)

            df['upperCI_epi'] = np.percentile(test_epi, 97.5)
            df['lowerCI_epi'] = np.percentile(test_epi, 2.5)

            return df


        def get_wndws_prc(df, flu_var, wnd_sz, start_date, end_date):
            g = df
            g['TestDate'] = pd.to_datetime(g['TestDate'])
            idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
            g = g.groupby('TestDate').sum()
            g.index = pd.DatetimeIndex(g.index)
            g = g.reindex(idx, fill_value=0)
            gg = g[flu_var].rolling(wnd_sz, min_periods=1, center=True).sum()

            if pd.to_datetime(gg.idxmax() - round(wnd_sz / 2)) < pd.to_datetime(start_date):
                diff = pd.to_datetime(start_date) - pd.to_datetime(gg.idxmax() - round(wnd_sz / 2))
                strt_wnd = gg.idxmax() - round(wnd_sz / 2) + diff
                end_wnd = gg.idxmax() + round(wnd_sz / 2) + diff
            elif pd.to_datetime(gg.idxmax() + round(wnd_sz / 2)) > pd.to_datetime(end_date):
                diff = pd.to_datetime(end_date) - pd.to_datetime(gg.idxmax() - round(wnd_sz / 2))
                strt_wnd = gg.idxmax() - round(wnd_sz / 2) + diff
                end_wnd = gg.idxmax() + round(wnd_sz / 2) + diff
            else:
                strt_wnd = gg.idxmax() - round(wnd_sz / 2)
                end_wnd = gg.idxmax() + round(wnd_sz / 2)

            ins = (g.index > strt_wnd) & (g.index < end_wnd)
            TotPos_epi = g[ins][flu_var].sum()
            TotPos_bsl = g[~ins][flu_var].sum()
            Total_Tests_bsl = g[~ins]['Total_Tests'].sum()
            Total_Tests_epi = g[ins]['Total_Tests'].sum()

            return strt_wnd, end_wnd, TotPos_epi, TotPos_bsl, Total_Tests_epi, Total_Tests_bsl

        qf = Filter(grp, start_date, end_date, n_filter, flu_var)
        df = qf.groupby(grp).apply(get_wndws_prc, flu_var=flu_var, wnd_sz=wnd_sz, start_date=start_date, end_date=end_date)
        df = df.apply(pd.Series)
        df.columns = ['strt_wnd', 'end_wnd', 'TotPos_epi', 'TotPos_bsl', 'Total_Tests_epi', 'Total_Tests_bsl']
        df = df.reset_index(drop=False)
        df[['strt_wnd', 'end_wnd']] = df[['strt_wnd', 'end_wnd']].astype('str')
        df = mk_shp(df, grp, 'EPA_SummerFlu/shp_out/Prc/prc')
        df['lat'] = df.geometry.y

        tmp = np.empty([0, 6])
        for i in np.arange(25, 55, .5):
            ins = (df['lat'] > i - 2) & (df['lat'] < i + 2)
            tmp = np.append(tmp, np.array([[i, ins.sum(), df[ins]['TotPos_bsl'].sum(), df[ins]['Total_Tests_bsl'].sum(),df[ins]['TotPos_epi'].sum(), df[ins]['Total_Tests_epi'].sum()]]), axis=0)

        df2 = pd.DataFrame(tmp, columns=['lat', 'n', 'TotPos_bsl', 'Total_Tests_bsl','TotPos_epi', 'Total_Tests_epi'])
        df2['prc_bsl'] = df2['TotPos_bsl'] / df2['Total_Tests_bsl'] * 100
        df2['prc_epi'] = df2['TotPos_epi'] / df2['Total_Tests_epi'] * 100
        df2 = df2.dropna()

        df3 = df2.apply(btstrp,axis=1)

        if pd.to_datetime(df['strt_wnd'].min()) < pd.to_datetime('2017-6-30'):
            color = 'C0'
        else:
            color = 'C1'

        vars = ['bsl', 'epi']

        for i in vars:
            ax1 = plt.axes()

            plt.fill_between(df3['lat'], df3['upperCI_'+i], df3['lowerCI_'+i], color=color, alpha=.15, linewidth=0.0)

            df3.plot(x='lat', y='prc_'+i, legend=False, color=color, ax=ax1)

            ax1.set_ylim([0, df3['prc_'+i].max() * 1.1])
            plt.axhline(y=df2['prc_'+i].median(), linestyle='--', color='k', linewidth=.25)
            plt.ylabel('% Positive')
            plt.xlabel('Latitude')
            fig = ax1.get_figure()

            file_out = flu_var + '_' + i + '_' + start_date[:4]
            f_out = 'EPA_SummerFlu/Figures/PercByLat/' + file_out + '.png'
            fig.savefig(f_out)
            plt.clf()

        return df


    flu_var = ['Pos_A', 'Pos_B']

    for var in flu_var:
        mk_df_prcs(grp, start_date1, end_date1, n_filter, var, wnd_sz)
        mk_df_prcs(grp, start_date2, end_date2, n_filter, var, wnd_sz)


def mk_supp_prct_maps(grp, start_date1, end_date1, start_date2, end_date2, n_filter, wnd_sz):


    def mk_df_prcs(grp, start_date, end_date, n_filter, flu_var, wnd_sz):


        def get_wndws_prc(df, flu_var, wnd_sz, start_date, end_date):
            g = df
            g['TestDate'] = pd.to_datetime(g['TestDate'])
            idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
            g = g.groupby('TestDate').sum()
            g.index = pd.DatetimeIndex(g.index)
            g = g.reindex(idx, fill_value=0)
            gg = g[flu_var].rolling(wnd_sz, min_periods=1, center=True).sum()

            if pd.to_datetime(gg.idxmax() - round(wnd_sz / 2)) < pd.to_datetime(start_date):
                diff = pd.to_datetime(start_date) - pd.to_datetime(gg.idxmax() - round(wnd_sz / 2))
                strt_wnd = gg.idxmax() - round(wnd_sz / 2) + diff
                end_wnd = gg.idxmax() + round(wnd_sz / 2) + diff
            elif pd.to_datetime(gg.idxmax() + round(wnd_sz / 2)) > pd.to_datetime(end_date):
                diff = pd.to_datetime(end_date) - pd.to_datetime(gg.idxmax() - round(wnd_sz / 2))
                strt_wnd = gg.idxmax() - round(wnd_sz / 2) + diff
                end_wnd = gg.idxmax() + round(wnd_sz / 2) + diff
            else:
                strt_wnd = gg.idxmax() - round(wnd_sz / 2)
                end_wnd = gg.idxmax() + round(wnd_sz / 2)

            ins = (g.index > strt_wnd) & (g.index < end_wnd)
            TotPos_epi = g[ins][flu_var].sum()
            TotPos_bsl = g[~ins][flu_var].sum()
            Total_Tests_bsl = g[~ins]['Total_Tests'].sum()
            Total_Tests_epi = g[ins]['Total_Tests'].sum()

            return strt_wnd, end_wnd, TotPos_epi, TotPos_bsl, Total_Tests_epi, Total_Tests_bsl

        qf = Filter(grp, start_date, end_date, n_filter, flu_var)
        df = qf.groupby(grp).apply(get_wndws_prc, flu_var=flu_var, wnd_sz=wnd_sz, start_date=start_date, end_date=end_date)
        df = df.apply(pd.Series)
        df.columns = ['strt_wnd', 'end_wnd', 'TotPos_epi', 'TotPos_bsl', 'Total_Tests_epi', 'Total_Tests_bsl']
        df = df.reset_index(drop=False)
        df[['strt_wnd', 'end_wnd']] = df[['strt_wnd', 'end_wnd']].astype('str')
        df = mk_shp(df, grp, 'EPA_SummerFlu/shp_out/Prc/prc')
        df['lat'] = df.geometry.y
        df['prc_bsl'] = df['TotPos_bsl'] / df['Total_Tests_bsl'] * 100
        df['prc_epi'] = df['TotPos_epi'] / df['Total_Tests_epi'] * 100
        fout = 'EPA_SummerFlu/shp_out/Prc/prc'+ '_' + flu_var + '_' + start_date
        df = mk_shp(df, grp, fout)


    flu_var = ['Pos_A', 'Pos_B']

    for var in flu_var:
        mk_df_prcs(grp, start_date1, end_date1, n_filter, var, wnd_sz)
        mk_df_prcs(grp, start_date2, end_date2, n_filter, var, wnd_sz)


def mk_Moran_plot(grp, flu_var, start_date, end_date, wnd_sz, n_filter):

    df = mk_final_df(grp, flu_var, start_date, end_date, wnd_sz, n_filter, 'epibase')
    df = df.to_crs({'proj': 'aea'})
    df = df.rename(columns={'centroid': 'geometry'})
    tmp = np.empty([1,0])
    for i in range(10):
        wt = ps.weights.DistanceBand.from_dataframe(df, threshold=i*100000, binary=True)
        wt.transform = 'r'
        mi = ps.Moran(np.log(df['ratio']), wt, two_tailed=False, permutations=1000)
        tmp = np.append(tmp,mi.I)
        print(tmp)

    plt.plot(range(10),tmp)
    plt.show()


def run_spatial_reg(grp, flu_var, start_date, end_date, wnd_sz, n_filter, hilo_epibase):

    def plot_btstrp(Y, X, W):
        for i in range(1000):
            ins = np.random.randint(len(X), size=(len(X), 1))
            y = Y[ins].reshape([len(X), 1])
            x = X[ins].reshape([len(X), 1])
            w = pd.Series(np.array(W)[ins].reshape([len(X), ]))
            lm = linear_model.LinearRegression()
            lm.fit(x, y, sample_weight=w)
            x = np.arange(X.min(), X.max(), .1)
            x = x.reshape([len(x), 1])
            y_pred = lm.predict(x)
            plt.plot(np.sort(x, 0), np.flipud(np.sort(10 ** y_pred, 0)), color='gray', linewidth=0.1, alpha=.05)

    df = mk_final_df_clim(grp, flu_var, start_date, end_date, wnd_sz, n_filter, hilo_epibase)
    df = df.to_crs({'proj': 'aea'})
    df = df.rename(columns={'centroid': 'geometry'})
    wt = ps.weights.DistanceBand.from_dataframe(df, threshold=600000, binary=True, silent=True)
    wt.transform = 'r'


    X_var = ['sh_flu', 'SH_bsl']
    Y = df[['log_ratio']].values
    X = df[[X_var]].values
    W = (df['epidemic'] + df['baseline']).astype(int) / 10

    ax = plt.axes()
    plot_btstrp(Y, X, W)

    m2 = ps.spreg.OLS(Y, X, w=wt, spat_diag=True, name_x=[X_var], name_y='log ratio')

    lm = linear_model.LinearRegression()
    lm.fit(X, Y, sample_weight=W)
    x = np.arange(X.min(), X.max(), .1)
    x = x.reshape([len(x), 1])
    y_pred = lm.predict(x)

    # plot data
    pd.DataFrame(df).plot.scatter(x=X_var, y='ratio', s=W, c='none', edgecolor='gray', marker="o",
                                  linewidth=1, alpha=0.5, ax=ax)
    plt.plot(np.sort(x, 0), np.flipud(np.sort(10 ** y_pred, 0)), color='r', linewidth=1)

    plt.ylabel('Cross-Seasonal Ratio')
    plt.xlabel('Specific Humidity (g/kg)')
    fig = ax.get_figure()

    file_out = flu_var + start_date[:4]
    f_out = 'EPA_SummerFlu/Figures/OLS_results/' + file_out + '.png'
    fig.savefig(f_out)
    plt.clf()

    # Run statsmodel regression
    wls_model = sm.WLS(Y, sm.add_constant(X), weights=W)
    results = wls_model.fit()
    print(results.summary())
    # rdf = pd.DataFrame([X_var, results.f_pvalue, results.rsquared, results.aic, results.params[0], results.params[1]])
    # f_out = 'EPA_SummerFlu/RegResults/'+X_var+'_'+start_date[:4]+'_'+flu_var[-2:]+'.csv'
    # rdf.to_csv()
    # plt.scatter(df[['log_ratio']].values,df[['sh_flu']].values,size=df['epidemic'])

    # print(m2.summary)


def mk_supp_tables(start_date1, end_date1, start_date2, end_date2, wnd_sz, n_filter, hilo_epibase):
    df0 = mk_final_df('mr', 'Pos_A', start_date1, end_date1, wnd_sz, n_filter, hilo_epibase)
    df0['Flu Type'] = 'A'
    df0['Year'] = '2016/2017'

    df1 = mk_final_df('mr', 'Pos_B', start_date1, end_date1, wnd_sz, n_filter, hilo_epibase)
    df1['Flu Type'] = 'B'
    df1['Year'] = '2016/2017'

    df2 = mk_final_df('mr', 'Pos_A', start_date2, end_date2, wnd_sz, n_filter, hilo_epibase)
    df2['Flu Type'] = 'A'
    df2['Year'] = '2017/2018'

    df3 = mk_final_df('mr', 'Pos_B', start_date2, end_date2, wnd_sz, n_filter, hilo_epibase)
    df3['Flu Type'] = 'B'
    df3['Year'] = '2017/2018'

    df = pd.concat([df0, df1, df2, df3])

    df = df.drop('centroid',1)

    df.to_csv(
        'EPA_SummerFlu/Figures/SuppTables/SuppTable1.csv')

    return df



####other utilityies

def SIR_1():

    #transmission rate
    k = 1/2 #average latency period
    gamma = 1/4  #mean infectious period
    sigma = 1/(1000)   #duration of immunity
    d = 1/(70*365) #human birth/death rate in days
    importRate = 1

    t0 = 0.0  #initial time in days
    tf = 4380 #number of days
    N = 1e6 #population size

    prop_susc = 0.6 #% proportion susceuptible
    prop_exp = 0.001 #% proportion susceuptible
    prop_inf = 0.0 #% proportion susceuptible
    prop_rec = 0.399 #% proportion susceuptible

    Sus_0 = N * prop_susc #susecptibles at t=0
    Exp_0 = N * prop_exp
    Inf_0 = N * prop_inf #incidence at t=0
    Rec_0 = N * prop_rec

    INPUT = (Sus_0, Inf_0, Exp_0, Rec_0, Inf_0)

    x = [0.0, 8.0, 12.0, 18.0, 23.0]
    y = [2, 1.5, 1, 1.3, 1.5]

    def beta_t(df,t):
        beta = df.minn_r0.iloc[round(t)]
        return beta


    def transform_sh_r0(x,y):
        df = pd.read_pickle(
            'EPA_SummerFlu/ClimateData/mrClimateData')
        df_miami = df[df['mr'] == 'Miami']
        df_minn = df[df['mr'] == 'Minneapolis']
        df_minn = df_minn.drop_duplicates(keep='first')
        df_miami = df_miami.drop_duplicates(keep='first')
        miami_SH = np.tile(df_miami['sh'].values, 8).tolist()
        minn_SH = np.tile(df_minn['sh'].values, 8).tolist()
        df = pd.DataFrame({'minn_sh': minn_SH, 'miami_sh': miami_SH})
        df['minn_sh'] = df['minn_sh'] * 1000
        df['miami_sh'] = df['miami_sh'] * 1000
        s = PchipInterpolator(x, y)
        df['miami_r0'] = s(df['miami_sh'].values)
        df['minn_r0'] = s(df['minn_sh'].values)
        df = df[['miami_r0','minn_r0']]*gamma
        return df


    def diff_eqs(t, x):
        '''The main set of equations'''
        beta = beta_t(df,t)
        print(sum(x[0:4]))
        Y = np.zeros(5)
        N = sum(x[0:4])
        Y[0] = d*N - beta * x[0] * x[2] / N + sigma*x[3] - d * x[0] - importRate #S
        Y[1] = beta * x[0] * x[2] / N - k * x[1] - d * x[1] + importRate #E
        Y[2] = k * x[1] - gamma * x[2] - d * x[2] #I
        Y[3] = gamma * x[2] - sigma * x[3] - d * x[3] #R
        Y[4] = k * x[1] #incidence
        return np.array(Y)  # For odeint


    def plot_ts(t,y,N):
        plt.subplot(311)
        plt.plot(t,y[:, 0]/sum(np.transpose(y[:,0:4])), '-g', label='Susceptibles')
        plt.plot(t,y[:, 1]/sum(np.transpose(y[:,0:4])), '-r', label='exposed')
        plt.plot(t,(y[:, 2]+y[:, 1])/sum(np.transpose(y[:,0:4])), '-y', label='i')

        plt.plot(t, y[:, 3]/sum(np.transpose(y[:,0:4])), '-k', label='Recovereds')
        plt.legend(loc=0)
        plt.vlines(np.arange(365,tf,365),0,np.max(y[:,0]/N),colors='gray')

        plt.title('Program_2_1.py')
        plt.xlabel('Time')
        plt.ylabel('Susceptibles and Recovereds')

        plt.subplot(312)
        inf = resample_data(t, y, t0, tf)
        plt.plot(np.arange(0,tf-1), inf/N,'-r', label='Infectious')
        plt.vlines(np.arange(0,tf,365),0,np.max(inf/N),colors='gray')

        plt.xlabel('Time')
        plt.ylabel('Infectious')
        plt.subplot(313)
        plt.plot(df.minn_r0/gamma, '-r', label='R0')
        plt.vlines(np.arange(0,tf,365),.5,df.miami_r0/gamma,colors='gray')


        plt.show()


    def resample_data(t, y, t0, tf):

        y6 = []
        y5 = np.diff(np.array(y[:, 4]))
        y5 = np.append(0, y5)

        for i in np.arange(t0, tf - 1):
            y6.append(sum(y5[((t >= i) & (t < i + 1))]))

        return np.array(y6)


    df = transform_sh_r0(x, y)

    solver = spi.ode(diff_eqs)
    solver.set_integrator('vode', method='bdf', nsteps=tf, order=45)
    solver.set_initial_value(INPUT, t0)

    y, t = [], []
    while solver.successful() and solver.t < tf:
        solver.integrate(tf, step=True)
        y.append(solver.y)
        t.append(solver.t)

    t = np.array(t)
    y = np.array(y)

    plot_ts(t, y, N)


def SIR_2():

    fixed_ratios = [50,250]

    mnths = np.arange(np.datetime64("2010-01-01"), np.datetime64("2010-12-31"), np.timedelta64(1, 'D'),
                                   dtype='datetime64[D]').astype('datetime64[D]')
    mnths = pd.DataFrame(np.transpose(np.tile(mnths, [1, 12])))
    mnths = mnths[0].dt.month
    smmr_mnths = mnths.isin([7, 8, 9])

    #transmission rate
    k = 1/2 #average latency period
    gamma = 1/4  #mean infectious period
    sigma = 1/(365)   #duration of immunity
    d = 1/(70*365) #human birth/death rate in days
    importRate = 1

    t0 = 0.0  #initial time in days
    tf = 4368 #number of days
    N = 1e6 #population size

    prop_susc = 0.6 #% proportion susceuptible
    prop_exp = 0.001 #% proportion susceuptible
    prop_inf = 0.0 #% proportion susceuptible
    prop_rec = 0.399 #% proportion susceuptible

    Sus_0 = N * prop_susc #susecptibles at t=0
    Exp_0 = N * prop_exp
    Inf_0 = N * prop_inf #incidence at t=0
    Rec_0 = N * prop_rec

    INPUT = (Sus_0, Inf_0, Exp_0, Rec_0, Inf_0)
    INPUT = np.tile(INPUT, [1, 2]).tolist()[0]

    x = [[0.0, 8.0, 10, 14.0, 18.0, 23.0],
         [0.0, 8.0, 12, 14.0, 18.0, 23.0],
         [0.0, 8.0, 12, 14.0, 18.0, 23.0]]

    y = [
         [2.5, 2.5, 1.3, 1.3, 1.3, 1.5],
         [2.2, 2, 1.5, 1, 1.1, 1.8],
         [2.2, 2, 1.5, 1, 1.1, 1.8],
         [2.5, 2.5, 1, 1.2, 1.3, 1.5],
         [2.5, 2.5, 1.5, 1.5, 1.6, 1.8]
         ]

    x = x[0]
    y = y[0]


    def beta_t(df,t):
        #beta = df[['minn_r0','miami_r0']].iloc[round(t)].values
        return  df[round(t),:]


    def transform_sh_r0(x,y):
        df = pd.read_pickle(
            'EPA_SummerFlu/ClimateData/mrClimateData')
        df_miami = df[df['mr'] == 'Miami']
        df_minn = df[df['mr'] == 'Minneapolis']
        df_minn = df_minn.drop_duplicates(keep='first')
        df_miami = df_miami.drop_duplicates(keep='first')
        miami_SH = np.tile(df_miami['sh'].values, 8).tolist()
        minn_SH = np.tile(df_minn['sh'].values, 8).tolist()
        df = pd.DataFrame({'minn_sh': minn_SH, 'miami_sh': miami_SH})
        df['minn_sh'] = df['minn_sh'] * 1000
        df['miami_sh'] = df['miami_sh'] * 1000
        s = PchipInterpolator(x, y)
        df['miami_r0'] = s(df['miami_sh'].values)
        df['minn_r0'] = s(df['minn_sh'].values)
        df = df[['miami_r0','minn_r0']]*gamma
        return df


    def diff_eqs(t, x):
        x = x.reshape([2, 5])

        beta = beta_t(df,t)

        Y = np.zeros([5,2])
        print(t)

        N = sum(np.transpose(x[:, 0:4]))
        Y[0,:] = d*N - beta * x[:,0] * x[:,2] / N + sigma*x[:,3] - d * x[:,0] - importRate #S
        Y[1,:] = beta * x[:,0] * x[:,2] / N - k * x[:,1] - d * x[:,1] + importRate #E
        Y[2,:] = k * x[:,1] - gamma * x[:,2] - d * x[:,2] #I
        Y[3,:] = gamma * x[:,2] - sigma * x[:,3] - d * x[:,3] #R
        Y[4,:] = k * x[:,1] #incidence

        return np.transpose(Y).reshape([10,]) # For odeint


    def score_estimates(mx,fixed_ratios,smmr_mnths,t0,tf,t):
        [inf_minn, inf_miami] = resample_data(t, mx, t0, tf)
        smmr_mnths = smmr_mnths[0:inf_minn.shape[0]]

        minn_ratio = sum(inf_minn[~smmr_mnths])/sum(inf_minn[smmr_mnths])
        miami_ratio = sum(inf_miami[~smmr_mnths])/sum(inf_miami[smmr_mnths])
        score = (np.log(minn_ratio) - np.log(fixed_ratios[0])) ** 2 + (np.log(miami_ratio) - np.log(fixed_ratios[1])) ** 2
        return score


    def resample_data(t, y, t0, tf):
        y6 = []
        y5 = np.diff(np.array(y[:, 4]))
        y5 = np.append(0, y5)

        for i in np.arange(t0, tf - 1):
            y6.append(sum(y5[((t >= i) & (t < i + 1))]))

        y7 = []
        y9 = np.diff(np.array(y[:, 9]))
        y9 = np.append(0, y9)

        for i in np.arange(t0, tf - 1):
            y7.append(sum(y9[((t >= i) & (t < i + 1))]))

        return np.array(y6), np.array(y7)


    def plot_ts(t,y,N):
        plt.subplot(311)
        plt.plot(t,y[:, 0]/sum(np.transpose(y[:,0:4])), '-r', label='Susceptibles')
        plt.plot(t,y[:, 3]/sum(np.transpose(y[:,0:4])), '-r', label='exposed')

        plt.plot(t, y[:, 5] / sum(np.transpose(y[:, 5:9])), '-k', label='Susceptibles')
        plt.plot(t, y[:, 8] / sum(np.transpose(y[:, 5:9])), '-k', label='exposed')
        plt.vlines(np.arange(365,tf,365),0,np.max(y[:,0]/N),colors='gray')
        plt.ylim([0,1])

        plt.subplot(312)
        [inf0, inf1] = resample_data(t, y, t0, tf)
        plt.plot(np.arange(0,tf-1), inf0/N,'-r', label='Infectious')
        plt.plot(np.arange(0,tf-1), inf1/N,'-k', label='Infectious')
        plt.vlines(np.arange(0,tf,365),0,np.max(inf1/N),colors='gray')

        plt.subplot(313)
        plt.plot(df.minn_r0/gamma, '-k', label='R0')
        plt.plot(df.miami_r0/gamma, '-r', label='R0')
        plt.vlines(np.arange(0, tf, 365), min(df.minn_r0 / gamma), max(df.minn_r0 / gamma), colors='gray')

        plt.show()


    df = transform_sh_r0(x, y).values
    #t_range = np.arange(t0,tf,1)

    #RES = spi.odeint(diff_eqs, INPUT, t_range)
    solver = spi.ode(diff_eqs)
    solver.set_integrator('vode', method='Adams', nsteps=tf, order=10, min_step=1e-5)
    solver.set_initial_value(INPUT, t0)

    mx, t = [], []
    while solver.successful() and solver.t < tf:
        solver.integrate(tf, step=True)
        mx.append(solver.y)
        t.append(solver.t)



    t = np.array(t)
    mx = np.array(mx)
    df = transform_sh_r0(x, y)

    score = score_estimates(mx, fixed_ratios, smmr_mnths, t0, tf, t)



    plot_ts(t, mx, N)


def mk_sh_ts():
    sh = pd.read_pickle('EPA_SummerFlu/ClimateData/mrClimateData')
















