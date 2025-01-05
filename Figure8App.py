#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 08:41:04 2024

@author: antoinemaratray
"""
import streamlit as st
import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from highlight_text import fig_text
from statsbombpy import sb
import gdown
import warnings
warnings.filterwarnings('ignore')

# StatsBomb API credentials
username = "admin@figure8.com"  
password = "QCOKgqp1"  

def load_data():
    base_url = "https://drive.google.com/uc?id="
    file_ids = {
        "consolidated_matches": "11F6TzXOTe2SgwYCiA2vooWs_6luGSY5w",
        "player_mapping_with_names": "1usGHXxhA5jX4u-H2lua0LyRvBljA1BIG",
        "mapping_matches": "1_Xqdfo69QfuV5aHHNw6omG_bZIU9xAcb",
        "mapping_players": "1jxZ9OzY376i2ac71Vxuyoke1pDa6PEHY",
        "wyscout_physical_data": "1pXB9Ih9gKZP3_i05X7QIW7Uw95AD8jTX",
        "player_stats": "1oExf9zGs-E-pu-Q0H9Eyo7-eqXue8e1Z",
    }

    import requests_cache
    requests_cache.install_cache("gdown_cache", backend="sqlite", expire_after=3600,backend_options={"timeout": 10})
    # Download datasets
    consolidated_matches = pd.read_csv(gdown.download(base_url + file_ids["consolidated_matches"], quiet=False))
    player_mapping_with_names = pd.read_csv(gdown.download(base_url + file_ids["player_mapping_with_names"], quiet=False))
    wyscout_physical_data = pd.read_csv(gdown.download(base_url + file_ids["wyscout_physical_data"], quiet=False))
    
    # Load player_stats as JSON
    player_stats_json = gdown.download(base_url + file_ids["player_stats"], quiet=False)
    with open(player_stats_json, 'r') as f:
        player_stats = pd.DataFrame(json.load(f))

    return consolidated_matches, player_mapping_with_names, player_stats, wyscout_physical_data
consolidated_matches, player_mapping_with_names, player_stats, wyscout_physical_data = load_data()

def get_events_from_statsbomb(match_id):
    # Use StatsBomb API to fetch event data
    events_df = sb.events(match_id=match_id, creds={"user": username, "passwd": password})
    return events_df


def generate_full_visualization(filtered_events, events_df, season_stats, match_id, player, wyscout_data, opponent, player_minutes):
    # Ensure valid locations in filtered events
    filtered_events = filtered_events[
        filtered_events['location'].apply(lambda loc: isinstance(loc, list) and len(loc) == 2)
    ]
    filtered_events[['x', 'y']] = pd.DataFrame(filtered_events['location'].tolist(), index=filtered_events.index)
    
    # Extract 'end_x', 'end_y' for passes and carries
    filtered_events[['end_x', 'end_y']] = filtered_events.apply(
        lambda row: pd.Series(row['pass_end_location'] if row['type'] == 'Pass' else 
                              (row['carry_end_location'] if row['type'] == 'Carry' else [None, None])), axis=1
    )
    
    # Retain Shot events for separate processing
    shot_events = filtered_events[filtered_events["type"] == "Shot"]
    
    # Extract shot-specific columns
    filtered_events['shot_outcome'] = filtered_events.apply(
        lambda row: row['shot_outcome'] if row['type'] == 'Shot' else None, axis=1
    )
    filtered_events['shot_statsbomb_xg'] = filtered_events.apply(
        lambda row: row['shot_statsbomb_xg'] if row['type'] == 'Shot' else None, axis=1
    )
    
    # Parse location for Shot events
    if not shot_events.empty:
        shot_events[['x', 'y']] = pd.DataFrame(shot_events['location'].tolist(), index=shot_events.index)

    # Set up the 2x4 grid for the plots
    fig = plt.figure(figsize=(20, 28))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[5, 5, 5, 6])
    gs.update(hspace=0.3)

    # Set up the 2x4 grid for the plots
    fig = plt.figure(figsize=(20, 28))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[5, 5, 5, 6])
    gs.update(hspace=0.3)

    # ********* Plot 1: Touches Heatmap *********
    ax1 = fig.add_subplot(gs[0, 0])
    pitch = Pitch(line_color='black', pitch_color='white')
    pitch.draw(ax=ax1)
    cmap = LinearSegmentedColormap.from_list("white_to_darkred", ["white", "#9d0208"])
    
    # Filter touches-related events
    touch_events = filtered_events[
        (filtered_events["type"].isin(["Pass", "Dribble", "Carry", "Ball Receipt"]))
    ]
    
    # Extract and plot
    touch_events[['x', 'y']] = pd.DataFrame(touch_events['location'].tolist(), index=touch_events.index)
    pitch.kdeplot(touch_events['x'], touch_events['y'], ax=ax1, cmap=cmap, fill=True, levels=100, alpha=0.7)
    ax1.set_title("Touches Heatmap", fontsize=16)

    # ********* Plot 2: Passes and Carries *********
    ax2 = fig.add_subplot(gs[0, 1])
    pitch.draw(ax=ax2)
    
    # Separate pass and carry data for plot customization
    filtered_events[['end_x', 'end_y']] = filtered_events.apply(
        lambda row: pd.Series(row['pass_end_location'] if row['type'] == 'Pass' else 
                              (row['carry_end_location'] if row['type'] == 'Carry' else [None, None])), axis=1
    )
    
    df_passes_completed = filtered_events[(filtered_events["type"] == "Pass") & (filtered_events['pass_outcome'].isna())]
    df_passes_incomplete = filtered_events[(filtered_events["type"] == "Pass") & (filtered_events['pass_outcome'].notna())]
    df_carries = filtered_events[filtered_events["type"] == "Carry"]
    
    # Plot lines
    pitch.lines(df_passes_completed['x'], df_passes_completed['y'], df_passes_completed['end_x'], df_passes_completed['end_y'],
                lw=5, transparent=True, comet=True, label='Completed Passes', color='#0a9396', ax=ax2)
    pitch.lines(df_passes_incomplete['x'], df_passes_incomplete['y'], df_passes_incomplete['end_x'], df_passes_incomplete['end_y'],
                lw=5, transparent=True, comet=True, label='Incomplete Passes', color='#ba4f45', ax=ax2)
    pitch.lines(df_carries['x'], df_carries['y'], df_carries['end_x'], df_carries['end_y'],
                lw=5, transparent=True, comet=True, linestyle='dotted', label='Carries', color='#fcbf49', ax=ax2)
    
    # Add legend and title
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, facecolor='white', handlelength=3, fontsize=13, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax2.set_title("Passes and Carries", fontsize=16)

    # ********* Plot 3: Defensive Actions *********
    defensive_events = filtered_events[
        filtered_events["type"].isin(["Pressure", "Block", "Interception", "Duel"])
    ]
    pressure_events = defensive_events[defensive_events["type"] == "Pressure"]
    other_events = defensive_events[defensive_events["type"] != "Pressure"]
    
    pressure_x = [loc[0] for loc in pressure_events['location'] if isinstance(loc, list)]
    pressure_y = [loc[1] for loc in pressure_events['location'] if isinstance(loc, list)]
    other_x = [loc[0] for loc in other_events['location'] if isinstance(loc, list)]
    other_y = [loc[1] for loc in other_events['location'] if isinstance(loc, list)]
    
    ax3 = fig.add_subplot(gs[1, 0])
    pitch.draw(ax=ax3)
    cmap = LinearSegmentedColormap.from_list("gray_to_darkblue", ["white", "#003566"])
    
    # Heatmap
    pitch.kdeplot(all_x := other_x + pressure_x, all_y := other_y + pressure_y, ax=ax3, cmap=cmap, fill=True, levels=100, alpha=0.6, zorder=1)
    pitch.scatter(other_x, other_y, ax=ax3, color="blue", edgecolors="white", alpha=0.8, s=50, label="Defensive Actions")
    pitch.scatter(pressure_x, pressure_y, ax=ax3, color="orange", edgecolors="white", alpha=0.8, s=50, marker='^', label="Pressure Events")
    ax3.legend(facecolor='white', loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax3.set_title("Defensive Actions", fontsize=16)

    # ********* Plot 4: Shot Map *********
    # Filter for shots by the selected player and exclude penalties
    player_shots = events_df[
        (events_df["type"] == "Shot") &
        (events_df["player"] == player) &
        (events_df["shot_type"] != "Penalty")
    ]
    
    # Ensure `location` is valid and parse it into `x`, `y`, `z`
    if not player_shots.empty and "location" in player_shots.columns:
        player_shots[['x', 'y', 'z']] = pd.DataFrame(player_shots['location'].tolist(), index=player_shots.index)
    
        # Separate non-goals and goals
        non_goals = player_shots[player_shots["shot_outcome"] != "Goal"]
        goals = player_shots[player_shots["shot_outcome"] == "Goal"]
    
        # Set up the subplot for the shot map
        ax_shot_map = fig.add_subplot(gs[1, 1])
        pitch = VerticalPitch(
            pitch_type='statsbomb',
            pitch_color='white',  # Background color for pitch
            line_color='black',  # Pitch lines color
            line_zorder=1,
            pad_bottom=-15,
            linewidth=2,
            half=True
        )
        pitch.draw(ax=ax_shot_map)
    
        # Plot non-goals
        if not non_goals.empty:
            pitch.scatter(
                non_goals['x'], 
                non_goals['y'], 
                s=non_goals['shot_statsbomb_xg'] * 1000, 
                c='lightgrey',  
                edgecolors='black', 
                marker='o', 
                alpha=0.8, 
                ax=ax_shot_map, 
                label='Non-Goals'
            )
    
        # Plot goals
        if not goals.empty:
            pitch.scatter(
                goals['x'], 
                goals['y'], 
                s=goals['shot_statsbomb_xg'] * 1000, 
                c='#40916c',  
                edgecolors='black', 
                marker='o', 
                alpha=0.8, 
                ax=ax_shot_map, 
                label='Goals'
            )
    
        # Add legend below the plot without a frame
        legend = ax_shot_map.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.1),
            title='Shot Outcome',
            frameon=False,
            ncol=2
        )
        for legend_handle in legend.legend_handles:
            legend_handle._sizes = [100]  # Adjust legend marker size
    
        # Add the title
        ax_shot_map.set_title(f'Shot Map', fontsize=20, color='black')

        
        # ********* Plot 5: Benchmarking - Match vs Season *********
    alternate_names = {
    "Anderson José Lopes de Souza": "Anderson Lopes"
    }    
        
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
    
    mapped_player = alternate_names.get(player, player)
    # Calculate total minutes played by each player
    season_stats['player_total_minutes'] = season_stats.groupby('player_name')['player_match_minutes'].transform('sum')
    
    # Filter league-wide players with >1000 minutes
    minutes_threshold = 1000
    league_stats = season_stats[season_stats['player_total_minutes'] > minutes_threshold]
    
    # Filter the selected player's matches
    player_matches = season_stats[season_stats['player_name'] == mapped_player]
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
    
    # Define colors consistent with the rest of the dashboard
    color4, color3, color2, color1 = '#0d1b2a', '#1b263b', '#415a77', '#778da9'
    cmap = LinearSegmentedColormap.from_list("custom_blue_gradient", [color1, color2, color3, color4])
    
    # Retrieve lineup data
    lineup_data = sb.lineups(match_id=match_id, creds={"user": username, "passwd": password})
    
    # Get the team data for the focus team and opponent
    if player in lineup_data[home_team]["player_name"].values:
        team_data = lineup_data[home_team]
        team_name = home_team
    else:
        team_data = lineup_data[away_team]
        team_name = away_team
    
    # Get the opponent's name
    opponent = home_team if team_name == away_team else away_team
    
    # Get the list of teammates for the selected team
    teammates = team_data["player_name"].tolist()

    
    # Filter events for passes made by the selected player to teammates
    df_passes = events_df[
        (events_df["player"] == player) &  # Filter for the selected player
        (events_df["type"] == "Pass") &  # Only passes
        (events_df["pass_outcome"].isna()) &  # Only successful passes
        (events_df["pass_recipient"].isin(teammates))  # Only passes to teammates
    ]
    
    # Group passes by player and recipient to calculate pass counts
    pass_matrix = df_passes.groupby(["player", "pass_recipient"]).size().unstack(fill_value=0)
    
    # Group passes by player and recipient to calculate pass counts
    pass_matrix = df_passes.groupby(["player", "pass_recipient"]).size().unstack(fill_value=0)
    
    # Sort pass_matrix by total passes received
    if not pass_matrix.empty:
        pass_matrix = pass_matrix.loc[:, pass_matrix.sum().sort_values(ascending=True).index]
    
        # Create the heatmap
        sns.heatmap(
            pass_matrix,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
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
    else:
        # Display a placeholder when pass_matrix is empty
        ax_pass_matrix.text(
            0.5, 0.5, "No Pass Data Available",
            ha='center', va='center', fontsize=16, color='gray', transform=ax_pass_matrix.transAxes
        )
        ax_pass_matrix.set_title(f"Pass Matrix for {player}", fontsize=18, pad=20)
        ax_pass_matrix.axis("off")  # Hide axes
    
    # Maintain aspect ratio and layout constraints
    ax_pass_matrix.set_aspect(aspect="auto")  # Adjust aspect ratio if needed
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
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
        st.stop()
    
    # Filter Wyscout data for the selected match and player
    filtered_wyscout_data = wyscout_physical_data[
        (wyscout_physical_data['matchId'] == wyscout_match_id) & (wyscout_physical_data['playerid'] == wyscout_player_id)
    ]
    
    # Ensure there is data for the match and player
    if filtered_wyscout_data.empty:
        st.error("No Wyscout data available for the selected player and match.")
        st.stop()
    
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
    
    # Populate metric data
    for metric, color in metrics_of_interest.items():
        metric_df = filtered_wyscout_data[filtered_wyscout_data['metric'] == metric]
        for _, row in metric_df.iterrows():
            if row['phase'] in hardcoded_phases:
                metric_data[metric][row['phase']] = row['value']
    
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
    max_speed_data = filtered_wyscout_data[filtered_wyscout_data['metric'] == "Max Speed"]
    for _, row in max_speed_data.iterrows():
        if row['phase'] in hardcoded_phases:
            bar_data[row['phase']] = row['value']
    
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


# Main Streamlit App
st.title("Figure 8: Post-Match Dashboard")

# Load Data
consolidated_matches, player_mapping_with_names, season_stats, wyscout_data = load_data()

# Sidebar Inputs
home_team = st.sidebar.selectbox("Select Home Team", consolidated_matches['home_team'].unique())
away_team = st.sidebar.selectbox("Select Away Team", consolidated_matches['away_team'].unique())
match_info = consolidated_matches[(consolidated_matches['home_team'] == home_team) & (consolidated_matches['away_team'] == away_team)]


if match_info.empty:
    st.error("No match found for the selected teams.")
else:
    match_id = match_info['statsbomb_id'].values[0]
    events_df = get_events_from_statsbomb(match_id)  # Fetch events using StatsBomb API
    players = events_df['player'].unique()  # Get unique player names from events
    player = st.sidebar.selectbox("Select Player (Start Typing Name)", players)

    if st.button("Generate Visualization"):
        with st.spinner("Generating plots..."):
            # Get opponent details
            lineups = sb.lineups(match_id=match_id, creds={"user": username, "passwd": password})
            opponent = [team for team in lineups.keys() if team != home_team][0]

            # Filter events for the selected player
            filtered_events = events_df[events_df['player'] == player]

            # Ensure the columns exist in player_stats
            if 'match_id' in player_stats.columns and 'player_name' in player_stats.columns:
                player_match = player_stats[
                    (player_stats['match_id'] == match_id) & 
                    (player_stats['player_name'] == player)
                ]
            else:
                st.error("The 'match_id' or 'player_name' column is missing in the player_stats data.")
                st.stop()

            # Extract player minutes
            player_minutes = player_match['player_match_minutes'].iloc[0] if not player_match.empty else 0

            # Generate and display visualizations
            fig = generate_full_visualization(
                filtered_events,
                events_df,
                player_stats,
                match_id,
                player,
                wyscout_physical_data,
                opponent,
                player_minutes
            )
            st.pyplot(fig)
