#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 08:41:04 2024

@author: antoinemaratray
"""
import requests
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsbombpy import sb
import streamlit as st
from mplsoccer import Pitch, VerticalPitch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from highlight_text import fig_text
import warnings
from requests.auth import HTTPBasicAuth
import os
warnings.filterwarnings('ignore')


base_url = "https://data.statsbomb.com/api/v8/events/"

# File IDs for Google Drive
file_ids = {
    "consolidated_matches": "11F6TzXOTe2SgwYCiA2vooWs_6luGSY5w",
    "player_mapping_with_names": "1usGHXxhA5jX4u-H2lua0LyRvBljA1BIG",
    "mapping_matches": "1_Xqdfo69QfuV5aHHNw6omG_bZIU9xAcb",
    "mapping_players": "1jxZ9OzY376i2ac71Vxuyoke1pDa6PEHY",
    "wyscout_physical_data": "1fqrtT1zqtFWBA8eYvIPSurUAvNhQGGXd",  # Your JSON file ID
    "player_stats": "1oExf9zGs-E-pu-Q0H9Eyo7-eqXue8e1Z",
}

def download_file_from_drive(file_id, destination_file_name):
    """Download a file from Google Drive using its ID."""
    url = f"{base_url}{file_id}&export=download"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(destination_file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {destination_file_name} from Google Drive.")
    else:
        print(f"Failed to download file from Google Drive. Status code: {response.status_code}")
        return None
    return destination_file_name

def load_json_file(file_name):
    """Load a JSON file and handle any issues."""
    try:
        with open(file_name, "r") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from {file_name}: {e}")
        return None

def fetch_events_from_statsbomb(match_id):
    """Fetch events for a given match from StatsBomb API using statsbombpy."""
    # Replace 'your_username' and 'your_password' with your actual StatsBomb credentials
    username = "admin@figure8.com"
    password = "QCOKgqp1"
    
    try:
        # Fetch events using statsbombpy
        events_df = sb.events(match_id=match_id, creds={"user": username, "passwd": password})
        return events_df
    except Exception as e:
        print(f"Error fetching events: {e}")
        return pd.DataFrame()
        
@st.cache_data
def load_data():
    # Initialize variables
    consolidated_matches = None
    player_mapping_with_names = None
    sb_events = None
    player_stats = None
    wyscout_physical_data = None

    for key, file_id in file_ids.items():
        file_name = f"{key}.csv" if key not in ["player_stats", "wyscout_physical_data"] else f"{key}.json"
        download_file_from_drive(file_id, file_name)

        if file_name.endswith('.csv'):
            if key == "consolidated_matches":
                consolidated_matches = pd.read_csv(file_name)
                st.write("Columns in consolidated_matches DataFrame:", consolidated_matches.columns) 
                st.write("First few rows of consolidated_matches DataFrame:", consolidated_matches.head())  
            elif key == "player_mapping_with_names":
                player_mapping_with_names = pd.read_csv(file_name)
        elif file_name.endswith('.json'):
            if key == "wyscout_physical_data":
                wyscout_physical_data = load_json_file(file_name)
            elif key == "player_stats":
                player_stats = load_json_file(file_name)

    return consolidated_matches, player_mapping_with_names, sb_events, player_stats, wyscout_physical_data

def filter_events(events_df):
    events_df = events_df[events_df["type"].isin([
        "Pass", "Dribble", "Carry", "Ball Receipt", "Block", "Interception", "Shot"
    ])]

    events_df = events_df[events_df['location'].apply(lambda loc: isinstance(loc, list) and len(loc) == 2)]

    events_df[['x', 'y']] = pd.DataFrame(events_df['location'].tolist(), index=events_df.index)
    return events_df
    
def generate_full_visualization(filtered_events, events_df, season_stats, match_id, player, wyscout_data, opponent, player_minutes):
    # Ensure valid locations in filtered events
    filtered_events = filtered_events[
        filtered_events['location'].apply(lambda loc: isinstance(loc, list) and len(loc) == 2)
    ]
    
    # Parse start location of events
    filtered_events[['x', 'y']] = pd.DataFrame(filtered_events['location'].tolist(), index=filtered_events.index)

    # Extract 'event_type' based on 'type' dictionary (assuming 'type' is always a dictionary with a 'name' key)
    filtered_events['event_type'] = filtered_events['type'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)

    # Extract 'end_x' and 'end_y' for passes and carries
    def get_end_location(row, event_type):
        if event_type == 'Pass' and isinstance(row.get('pass.end_location'), list):
            return row['pass.end_location'][0], row['pass.end_location'][1]
        elif event_type == 'Carry' and isinstance(row.get('carry.end_location'), list):
            return row['carry.end_location'][0], row['carry.end_location'][1]
        else:
            return None, None

    # Apply the get_end_location function to get 'end_x' and 'end_y' values
    filtered_events[['end_x', 'end_y']] = filtered_events.apply(
        lambda row: pd.Series(get_end_location(row, row['event_type'])), axis=1
    )
    
    # Set up the 2x4 grid for the plots
    fig = plt.figure(figsize=(20, 28))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[5, 5, 5, 6])
    gs.update(hspace=0.3)

    # ********* Plot 1: Touches Heatmap *********
    ax1 = fig.add_subplot(gs[0, 0])
    pitch = Pitch(line_color='black', pitch_color='white')
    pitch.draw(ax=ax1)
    cmap = LinearSegmentedColormap.from_list("white_to_darkred", ["white", "#9d0208"])
    pitch.kdeplot(filtered_events['x'], filtered_events['y'], ax=ax1, cmap=cmap, fill=True, levels=100, alpha=0.7)
    ax1.set_title("Touches Heatmap", fontsize=16)

    # ********* Plot 2: Passes and Carries *********
    ax2 = fig.add_subplot(gs[0, 1])
    pitch.draw(ax=ax2)
    df[['end_x', 'end_y']] = df.apply(
    lambda row: pd.Series(
        row['pass_end_location'] 
        if row['type'] == 'Pass' 
        else (row['carry_end_location'] if row['type'] == 'Carry' else [None, None])), axis=1)
    df_passes_completed = df[(df["type"] == "Pass") & (df['pass_outcome'].isna())]
    df_passes_incomplete = df[(df["type"] == "Pass") & (df['pass_outcome'].notna())]
    df_carries = filtered_events[filtered_events["event_type"] == "Carry"]
    pitch.lines(df_passes_completed['x'], df_passes_completed['y'], df_passes_completed['end_x'], df_passes_completed['end_y'],
                lw=5, transparent=True, comet=True, label='Completed Passes', color='#0a9396', ax=ax2)
    pitch.lines(df_passes_incomplete['x'], df_passes_incomplete['y'], df_passes_incomplete['end_x'], df_passes_incomplete['end_y'],
                lw=5, transparent=True, comet=True, label='Incomplete Passes', color='#ba4f45', ax=ax2)
    pitch.lines(df_carries['x'], df_carries['y'], df_carries['end_x'], df_carries['end_y'],
                lw=5, transparent=True, comet=True, linestyle='dotted', label='Carries', color='#fcbf49', ax=ax2)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, facecolor='white', handlelength=3, fontsize=13, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax2.set_title("Passes and Carries", fontsize=16)

    # ********* Plot 3: Defensive Actions *********
    defensive_events = filtered_events[
        (filtered_events['event_type'].isin(['Pressure', 'Block', 'Dribbled Past', 'Ball Recovery', 'Interception'])) |
        ((filtered_events['event_type'] == 'Duel') & (filtered_events['duel.type.name'] == 'Tackle'))
    ]
    pressure_events = defensive_events[defensive_events['event_type'] == 'Pressure']
    other_events = defensive_events[defensive_events['event_type'] != 'Pressure']
    pressure_x = pressure_events['x'].tolist()
    pressure_y = pressure_events['y'].tolist()
    other_x = other_events['x'].tolist()
    other_y = other_events['y'].tolist()
    ax3 = fig.add_subplot(gs[1, 0])
    pitch.draw(ax=ax3)
    cmap = LinearSegmentedColormap.from_list("gray_to_darkblue", ["white", "#003566"])
    pitch.kdeplot(all_x := other_x + pressure_x, all_y := other_y + pressure_y, ax=ax3, cmap=cmap, fill=True, levels=100, alpha=0.6, zorder=1)
    pitch.scatter(other_x, other_y, ax=ax3, color="blue", edgecolors="white", alpha=0.8, s=50, label="Defensive Actions")
    pitch.scatter(pressure_x, pressure_y, ax=ax3, color="orange", edgecolors="white", alpha=0.8, s=50, marker='^', label="Pressure Events")
    ax3.legend(facecolor='white', loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax3.set_title("Defensive Actions", fontsize=16)

    # ********* Plot 4: Shot Map *********
    shot_events = events_df[
        (events_df['event_type'] == 'Shot') & (events_df['match_id'] == match_id) & (events_df['player.name'] == player)
    ]
    shot_events = shot_events[shot_events['location'].apply(lambda loc: isinstance(loc, list) and len(loc) == 3)]
    if not shot_events.empty:
        shot_events[['x', 'y']] = pd.DataFrame(shot_events['location'].apply(lambda loc: loc[:2]).tolist(), index=shot_events.index)
        goals = shot_events[shot_events['shot.outcome.name'] == 'Goal']
        non_goals = shot_events[shot_events['shot.outcome.name'] != 'Goal']
        ax4 = fig.add_subplot(gs[1, 1])
        pitch = VerticalPitch(pitch_color='white', line_color='black', line_zorder=2, half=True)
        pitch.draw(ax=ax4)
        if not non_goals.empty:
            pitch.scatter(non_goals['x'], non_goals['y'], s=non_goals.get('shot.statsbomb_xg', 100) * 1000, edgecolor='black',
                          color='lightgrey', alpha=0.8, label='Non-Goals', ax=ax4)
        if not goals.empty:
            pitch.scatter(goals['x'], goals['y'], s=goals.get('shot.statsbomb_xg', 100) * 1000, edgecolor='black',
                          color='green', alpha=0.8, label='Goals', ax=ax4)
        ax4.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, title='Shot Outcome')
        ax4.set_title("Shot Map", fontsize=16)
        
        # ********* Plot 5: Benchmarking - Match vs Season *********
    # Define colors
    game_color = "#034694"
    bgcolor = "white"
    color1, color2, color3, color4 = '#0d1b2a', '#1b263b', '#415a77', '#778da9'
    metric_colors = {key: color for key, color in zip(
        [
            'player_match_successful_aerials', 'player_match_interceptions', 'player_match_ball_recoveries', 'player_match_tackles', 
            'player_match_pressures', 'player_match_aggressive_actions', 'player_match_counterpressures', 'player_match_op_passes', 
            'player_match_through_balls', 'player_match_crossing_ratio', 'player_match_dribbles', 'player_match_xa', 
            'player_match_np_shots', 'player_match_np_xg'
        ],
        [color1, color1, color1, color1, color2, color2, color2, color3, color3, color3, color3, color4, color4, color4]
    )}
    
    metrics_descriptive_names = [
        'Aerials won', 'Interceptions', 'Ball Recoveries', 'Tackles', 'Pressures',
        'Aggressive Actions', 'Counterpressures', 'OP Passes', 'Through Balls',
        'Crossing Ratio', 'Dribbles', 'xA', 'NP Shots', 'xG'
    ]
    
    # Calculate total minutes played by each player
    season_stats['player_total_minutes'] = season_stats.groupby('player_name')['player_match_minutes'].transform('sum')
    
    # Filter league-wide players with >1000 minutes
    minutes_threshold = 1000
    league_stats = season_stats[season_stats['player_total_minutes'] > minutes_threshold]
    
    # Filter the selected player's matches
    player_matches = season_stats[season_stats['player_name'] == player]
    player_match = player_matches[player_matches['match_id'] == match_id]
    if player_match.empty:
        raise ValueError(f"No data found for player {player} in match {match_id}")
    
    # Aggregating the player's season averages
    player_season_averages = player_matches.mean(numeric_only=True)
    
    # Metrics for analysis
    metric_mapping = {
        'player_match_successful_aerials': 'Aerials won',
        'player_match_interceptions': 'Interceptions',
        'player_match_ball_recoveries': 'Ball Recoveries',
        'player_match_tackles': 'Tackles',
        'player_match_pressures': 'Pressures',
        'player_match_aggressive_actions': 'Aggressive Actions',
        'player_match_counterpressures': 'Counterpressures',
        'player_match_op_passes': 'OP Passes',
        'player_match_through_balls': 'Through Balls',
        'player_match_crossing_ratio': 'Crossing Ratio',
        'player_match_dribbles': 'Dribbles',
        'player_match_xa': 'xA',
        'player_match_np_shots': 'NP Shots',
        'player_match_np_xg': 'xG',    
    }
    
    # Function to calculate percentile rank
    def calculate_percentile_rank(metric, value):
        all_values = league_stats[metric].dropna()
        if len(all_values) == 0:
            return 0
        return np.sum(all_values <= value) / len(all_values) * 100
    
    # Create subplot for benchmarking plot
    ax_benchmark = fig.add_subplot(gs[2, 0])
    fig.patch.set_facecolor(bgcolor)
    
    # Loop through each metric
    for i, (match_metric, metric_name) in enumerate(metric_mapping.items()):
        # Get player season average and match performance
        player_season_avg = player_season_averages.get(match_metric, 0)
        match_value = player_match[match_metric].iloc[0] if match_metric in player_match.columns else 0
    
        # Calculate percentiles
        season_percentile = calculate_percentile_rank(match_metric, player_season_avg)
        match_percentile = calculate_percentile_rank(match_metric, match_value)
    
        # Handle 0 values to ensure proper display
        if match_value == 0:
            match_percentile = 0  # Plot slightly to the left of the x-axis
    
        # Bar: Season average percentile
        ax_benchmark.barh(
            metric_name, season_percentile,
            color=metric_colors.get(match_metric, 'grey'),
            alpha=0.7, height=0.8
        )
    
        # Circle: Match percentile
        ax_benchmark.scatter(
            x=match_percentile, y=i,
            s=1000, color=bgcolor, edgecolor=game_color, lw=3, zorder=3
        )
        
        # Add the match value inside the circle
        ax_benchmark.text(
            match_percentile, i, f"{match_value:.2f}",
            va="center", ha="center", color=game_color, fontweight="bold", fontsize=10, zorder=10
        )
    
        # Add season average inside the bar
        ax_benchmark.text(
            season_percentile - 5, i, f"{player_season_avg:.2f}",
            va="center", ha="right", color="white", fontweight="bold", fontsize=10, zorder=10
        )

    
    # Styling and formatting
    ax_benchmark.set_yticks(range(len(metric_mapping)))
    ax_benchmark.set_yticklabels(metrics_descriptive_names, fontsize=14)
    ax_benchmark.set_xlabel('Percentile Ranking \n Match (circles) v Season Average (bars)', fontsize=16, fontweight="bold")
    ax_benchmark.axvline(50, color="grey", linestyle="--", linewidth=2)
    ax_benchmark.set_xlim(-10, 104)
    ax_benchmark.spines['top'].set_visible(False)
    ax_benchmark.spines['right'].set_visible(False)
    ax_benchmark.spines['left'].set_visible(False)
    ax_benchmark.spines['bottom'].set_visible(False)
    ax_benchmark.yaxis.grid(True, color="grey", alpha=0.5)
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
    # ********* Plot 6: Pass Matrix *********
    # Define Pass Matrix Plot
    ax_pass_matrix = fig.add_subplot(gs[2, 1])
    
    # Filter successful passes by the selected player in the chosen match
    color4, color3, color2, color1 = '#0d1b2a', '#1b263b', '#415a77', '#778da9'
    cmap = LinearSegmentedColormap.from_list("custom_blue_gradient", [color1, color2, color3, color4])
    
    df_passes = events_df[
        (events_df['match_id'] == match_id) &
        (events_df['player.name'] == player) &
        (events_df['type.name'] == "Pass") &
        (events_df['pass.outcome.name'].isna())  # Only successful passes
    ]
    
    # Group passes by recipient and count
    pass_matrix = df_passes.groupby("pass.recipient.name").size().reset_index(name='count')
    pass_matrix = pass_matrix.sort_values(by="count", ascending=True)
    
    # Set recipient names and counts for the heatmap
    recipient_names = pass_matrix["pass.recipient.name"]
    counts = pass_matrix["count"]
    
    # Create the Pass Matrix Heatmap
    sns.heatmap(
        data=[counts],
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        xticklabels=recipient_names,
        yticklabels=[''],
        ax=ax_pass_matrix,
        annot_kws={"size": 18, "color": "white", "weight": "bold"}
    )
    
    # Adjust labels and formatting
    ax_pass_matrix.set_title(f"Pass Matrix for {player}", fontsize=18, pad=20)
    ax_pass_matrix.set_xlabel("")
    ax_pass_matrix.set_ylabel("")
    ax_pass_matrix.tick_params(axis="x", labelrotation=45, labelsize=10)
    ax_pass_matrix.tick_params(axis="y", left=False)  # Hide y-axis ticks
    ax_pass_matrix.set_xticklabels(ax_pass_matrix.get_xticklabels(), ha='center')
    for label in ax_pass_matrix.get_xticklabels():
        label.set_ha('right')  # Align to the right
        label.set_position((label.get_position()[0] + 0.5, label.get_position()[1]))  # Adjust position slightly
    
    ax_pass_matrix.set_aspect(aspect="auto")  # Adjust aspect ratio if needed
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])  # Constrain layout to leave margins

    # ********* Plot 7: High-Speed Running *********
    
    # Map StatsBomb match and player IDs to Wyscout IDs
    wyscout_match_id = consolidated_matches.loc[consolidated_matches['statsbomb_id'] == match_id, 'wyscout_id']
    wyscout_player_id = player_mapping_with_names.loc[player_mapping_with_names['player_name'] == player, 'wyscout_id']
    
    # Ensure the IDs exist and extract their values
    if not wyscout_match_id.empty and not wyscout_player_id.empty:
        wyscout_match_id = wyscout_match_id.values[0]
        wyscout_player_id = wyscout_player_id.values[0]
    else:
        st.error("Unable to find Wyscout IDs for the selected match and player. Check your mappings.")
        
    ax_hsr = fig.add_subplot(gs[3, 0])
    
    # Metrics of interest and colors
    metrics_of_interest = {
        "High Speed Running (HSR) Distance": "#669bbc",
        "Sprinting Distance": "#1b263b",
        "High Intensity (HI) Distance": "#9e2a2b",
    }
    
    # Hardcoded x-axis phases
    hardcoded_phases = ["1'-15'", "16'-30'", "31'-45+'", "46'-60'", "61'-75'", "76'-120+'"]
    
    # Initialize metric data
    metric_data = {metric: {phase: 0 for phase in hardcoded_phases} for metric in metrics_of_interest}
    
    # Filter and map Wyscout data
    filtered_wyscout_data = [
        entry for entry in wyscout_data
        if entry['matchId'] == str(wyscout_match_id) and entry['playerid'] == str(wyscout_player_id)
    ]
    for metric, color in metrics_of_interest.items():
        for entry in filtered_wyscout_data:
            if entry['metric'] == metric and entry['phase'] in hardcoded_phases:
                metric_data[metric][entry['phase']] = float(entry['value'])
    
    # Truncate phases based on player minutes
    filtered_phases = [phase for phase in hardcoded_phases if int(phase.split('-')[0][:-1]) <= player_minutes]
    metric_data = {metric: {phase: metric_data[metric][phase] for phase in filtered_phases} for metric in metrics_of_interest}
    
    # Plot metrics
    for metric, color in metrics_of_interest.items():
        phases = filtered_phases
        values = [metric_data[metric][phase] for phase in phases]
        ax_hsr.plot(phases, values, marker='o', label=metric, color=color)
    
    # Format the plot
    ax_hsr.set_xticks(range(len(filtered_phases)))
    ax_hsr.set_xticklabels(filtered_phases, rotation=45, fontsize=10)
    ax_hsr.set_yticks(range(0, 251, 50))  # Adjust as needed
    ax_hsr.set_title("", fontsize=16, pad=20)
    ax_hsr.set_xlabel("", fontsize=12)
    ax_hsr.set_ylabel("Distance (m)", fontsize=12)
    ax_hsr.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=10, frameon=False)
    ax_hsr.grid(alpha=0.5)
    ax_hsr.spines['top'].set_visible(False)
    ax_hsr.spines['right'].set_visible(False)
    
    # ********* Plot 8: Max Speed Bar Chart *********
    ax_max_speed = fig.add_subplot(gs[3, 1])
    
    # Initialize bar data
    bar_data = {phase: 0 for phase in hardcoded_phases}
    
    # Populate bar data with Max Speed metric
    for entry in filtered_wyscout_data:
        if entry['metric'] == "Max Speed" and entry['phase'] in hardcoded_phases:
            bar_data[entry['phase']] = float(entry['value'])
    
    # Adjust bar data for filtered phases
    bar_data = {phase: bar_data[phase] for phase in filtered_phases}
    
    # Prepare data for the bar chart
    values = [bar_data[phase] for phase in filtered_phases]
    
    # Plot the bar chart
    if values:  # Check if there is data to plot
        bars = ax_max_speed.bar(filtered_phases, values, color="#003049", edgecolor="black")
        for bar, value in zip(bars, values):
            ax_max_speed.text(
                bar.get_x() + bar.get_width() / 2,  # Center of the bar
                bar.get_height() - 0.5,  # Slightly below the top of the bar
                f"{value:.1f}",  # Label text (formatted to 1 decimal place)
                ha="center",  # Horizontal alignment
                va="top",  # Vertical alignment
                color="white",  # Label color
                fontsize=12,  # Font size
                weight="bold"  # Bold text
            )
    else:
        ax_max_speed.text(
            0.5, 0.5, "No Max Speed Data Available", ha="center", va="center",
            fontsize=16, color="gray", transform=ax_max_speed.transAxes
        )
    
    
    # Format the chart
    ax_max_speed.set_title("", fontsize=16, pad=20)
    ax_max_speed.set_xticks(range(len(filtered_phases)))
    ax_max_speed.set_xticklabels(filtered_phases, rotation=45, fontsize=10)
    ax_max_speed.set_ylabel("Max Speed (km/h)", fontsize=12)  # Adjust as needed
    ax_max_speed.grid(axis="y", linestyle="--", alpha=0.7)
    ax_max_speed.spines['top'].set_visible(False)
    ax_max_speed.spines['right'].set_visible(False)
    
    fig_text(
        s=f"<{player}> \nvs <{opponent}> | Mins played: {player_minutes:.2f}",
        x=0.150, y=0.975, fontsize=30, 
        highlight_textprops=[{"fontweight": "bold"}, {"color": game_color, "fontweight": "bold"}]
    )

    return fig

st.title("Figure 8: Post-Match Dashboard")

# Load Data
consolidated_matches, player_mapping_with_names, sb_events, season_stats, wyscout_data = load_data()

# Sidebar Inputs with unique keys
home_team_key = f"home_team_select"
away_team_key = f"away_team_select"
player_select_key = f"player_select"

home_team = st.sidebar.selectbox("Select Home Team", consolidated_matches['home_team'].unique(), key=home_team_key)
away_team = st.sidebar.selectbox("Select Away Team", consolidated_matches['away_team'].unique(), key=away_team_key)
match_info = consolidated_matches[(consolidated_matches['home_team'] == home_team) & (consolidated_matches['away_team'] == away_team)]

if match_info.empty:
    st.error("No match found for the selected teams.")
else:
    match_id = match_info['statsbomb_id'].values[0]
    sb_events = fetch_events_from_statsbomb(match_id)  # Fetch events using statsbombpy
    
    if sb_events.empty:
        st.error("No events found for the selected match.")
    else:
        # After events are fetched, you can inspect them
        st.write("Columns in sb_events DataFrame:", sb_events.columns)
        st.write("First few rows of sb_events DataFrame:", sb_events.head())

        # Ensure the events contain player names
        sb_events['player_name'] = sb_events['player']  # Ensure 'player_name' column exists

        # Extract player names from the events for the selected match
        players = sb_events['player_name'].dropna().unique()  # List of player names from the events
        player = st.sidebar.selectbox("Select Player (Start Typing Name)", players, key=player_select_key)  # Dropdown for player selection

        # Filter the events for the selected player
        filtered_events = filter_events(sb_events[sb_events['player_name'] == player])

        # Generate and display the player's match dashboard visualization
        fig = generate_full_visualization(filtered_events, sb_events, season_stats, match_id, player, wyscout_data, home_team, 90)  # Example for 90 minutes played
        st.pyplot(fig)
