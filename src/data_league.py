import os
import time

import pandas as pd
from understatapi import UnderstatClient

from src.data import SEASONS, download_player_shots

def get_top_shooter_ids(max_per_season=30, pause=1.0):
    """Ids of the players with the most shots in each Serie A season."""
    ids=set()
    with UnderstatClient() as understat:
        for season in SEASONS:
            players=understat.league(league="Serie_A").get_player_data(season=season)
            top=sorted(players, key=lambda p: int(p ["shots"]), reverse=True)
            ids.update(str(p["id"]) for p in top[:max_per_season])
            time.sleep(pause)
        return ids

if __name__=="__main__":
    ids =get_top_shooter_ids()
    print(len(ids), "players to download")

    frames = []
    for i, player_id in enumerate(sorted(ids), start=1):
        try:
            frames.append(download_player_shots(player_id))
        except Exception as error:
            print("Skipped", player_id, "-", error)
        print (f"{i}/{len(ids)}")
        time.sleep(1.0)  

    league=pd.concat(frames, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    league.to_pickle("data/shots_league.pkl")
    print("Saved", len(league), "shots in data/shots_league.pkl")