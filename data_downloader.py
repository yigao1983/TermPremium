import datetime as dt
import pandas as pd


def fetch_yld_data(tenor_lst, start_date, end_date, save=True):
    try:
        yld_df = pd.read_csv('fed_nss_yld_{}_{}.csv'.format(start_date, end_date), index_col='Date')
    except IOError:
        from analytics.emp_analytics.USTreasuryYieldCurveModels import FedReserveNelsonSiegelSvenssonCurve
        yld_lst = []

        for tenor in tenor_lst:
            yld_series = FedReserveNelsonSiegelSvenssonCurve.yield_curve(tenor).values(start_date, end_date)
            yld_series.name = str(int(12 * tenor)) + 'M' if tenor < 1 else str(int(tenor)) + 'Y'
            yld_lst.append(yld_series)

        yld_df = pd.DataFrame(pd.concat(yld_lst, axis=1))

        if save:
            output_file = 'fed_nss_yld_{}_{}.csv'.format(start_date, end_date)
            yld_df.to_csv(output_file)

    return yld_df


if __name__ == '__main__':
    tenor_lst = [0.25, 0.5, 1, 2, 4, 7, 10]
    start_date = dt.date(1990, 1, 1)
    end_date = dt.date(2003, 12, 31)

    fetch_yld_data(tenor_lst, start_date, end_date, True)
