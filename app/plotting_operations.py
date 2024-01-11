import sqlite3
import pandas as pd
import plotly.express as px
import plotly.io as pio
from .create_database import parse_bit_mask,parse_single_bit_mask
from .database_operations import db_to_pandas
import numpy as np 
import os
import sys 
from tqdm import tqdm 
from matplotlib import pyplot as plt
def select_feed(df,feed):
    feed_position = df['feeds'].apply(lambda x: x.tolist().index(feed) if feed in x else None)
    selected_data = pd.DataFrame()

    # Apply the filter for each column
    for col in df.columns:
        if col == 'feeds':
            selected_data[col] = [feed if pos is not None else None for pos in feed_position]
        else:
            if isinstance(df[col].iloc[0], list):
                selected_data[col] = [row_data[pos] if pos is not None else None for pos, row_data in zip(feed_position, df[col])]
            else:
                selected_data[col] = [row_data if pos is not None else None for pos, row_data in zip(feed_position, df[col])]
    return selected_data

def plot_stat_vs_mjd(df, stat_column, feed, _figure_path):
    """Time line figures for each feed"""

    figure_path = os.path.join(_figure_path, 'timelines', f'Feed{feed:02d}')
    os.makedirs(figure_path, exist_ok=True)

    selected_data = select_feed(df,feed)

    special_sources = ['TauA','CasA','CygA','jupiter']
    selected_data['marker'] = selected_data['source_name'].apply(lambda x: 'circle' if x in special_sources else 'diamond')
    selected_data['date'] = df.index 
    selected_data.set_index('date', inplace=True)

    # Group all sources with Field in the name into one group called Galactic 
    selected_data['source_name'] = selected_data['source_name'].apply(lambda x: 'Galactic' if 'Field' in x else x)


    fig = px.scatter(selected_data, 
                    x=np.array(selected_data.index), 
                    y=stat_column, 
                    color="source_name",
                        symbol='marker',  
                    labels={"mjd": "MJD", stat_column: stat_column})
    # Set the x label to 'Date'
    fig.update_xaxes(title_text='Date')

    # Make the figure longer in the x direction 
    fig.update_layout(height=600, width=1200)

    # Set the y limits to be between the 2 and 98 percentiles
    vmin, vmax= np.nanpercentile(selected_data[stat_column],2),np.nanpercentile(selected_data[stat_column],98)
    vrange = vmax - vmin
    vmin -= 0.1*vrange
    vmax += 0.1*vrange
    fig.update_yaxes(range=[vmin,vmax])
    pio.write_html(fig, file=f'{figure_path}/Feed{feed:02d}_{stat_column}.html')
    pio.write_image(fig,file=f'{figure_path}/Feed{feed:02d}_{stat_column}.png')

def plot_bar_chart_observation_counts(_df, feed, _figure_path):

    figure_path = os.path.join(_figure_path, 'observation_counts')
    os.makedirs(figure_path, exist_ok=True)
    df = select_feed(_df,feed)

    unique_sources = df['source_name'].unique()
    unique_bad_values = [0,1,2,3,4,5,6,7,8,9]
    Nbad = len(unique_bad_values)
    # Set up the plot

    # Set up a list of colors for the bars for better visualization (optional)
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow','purple']

    # 0 = good, 1 = bad stats, 2 = bad instrument, 3 = bad pointing, 4 = no stats, 5 = no skydip, 6 = no vane, 7 = no source fit
    tick_names = ['good', 'bad stats', 'bad instrument', 'bad pointing', 'no stats', 'no skydip', 'no vane', 'no source fit', 'other','other_again']
    for i, source in enumerate(unique_sources):
        fig, ax = plt.subplots()
        total_observations = len(df[df['source_name'] == source])


        bad_values_count = [0] * Nbad
        source_df = df[df['source_name'] == source]

        bad_observations = 0 
        for index, row in source_df.iterrows():
            powers = parse_single_bit_mask(row['bad_observation'])
            for power in powers:
                bad_values_count[power] += 1
            if any([power in [1,2,3,4,6,7,8,9] for power in powers]):
                bad_observations += 1

        good_observations = total_observations - bad_observations
        bad_values_count[0] = good_observations
        #bad_values_count[0] -= (bad_values_count[1]+bad_values_count[2]+bad_values_count[3])
        bars = ax.bar(np.arange(Nbad), bad_values_count, color=colors, label=source)
        
        # Add numbers on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom')
        
        # Draw the horizontal line representing total observations
        ax.hlines(total_observations, xmin=0, xmax=Nbad+1, colors='black', linestyles='dashed')
        ax.text((2.5), total_observations + 0.2, f'Total: {total_observations}', ha='center', va='bottom')

        ax.set_xlabel('Bad Observation Value')
        ax.set_ylabel('Number of Observations')
        ax.set_title(f'Feed {feed:02d} Bad Observations for {source}')
        ax.set_xticks(np.arange(Nbad))
        ax.set_xticklabels(tick_names,rotation=45)
        plt.savefig(f'{figure_path}/{source}_Feed{feed:02d}_bad_observations.png')
        print('Figure name',f'{figure_path}/{source}_Feed{feed:02d}_bad_observations.png')
        plt.close()
        

def create_plots(database_name, figure_path):

    os.makedirs(figure_path, exist_ok=True)      

    df = db_to_pandas(database_name)

    #for feed in tqdm(range(1,20)):
    #    plot_bar_chart_observation_counts(df, feed, figure_path) 
    stat_names = ['tsys', 'gain', 'ra_offset', 'dec_offset', 'flux_band0','flux_band1','flux_band2','flux_band3', 'cal_factor', 'sky_brightness', 'bad_observation', 'auto_rms', 'red_noise', 'white_noise', 'spectrum_noise']

    # Plot stats vs. time
    for stat_name in stat_names:
        for feed in range(1,20):
            plot_stat_vs_mjd(df, stat_name, feed, figure_path)