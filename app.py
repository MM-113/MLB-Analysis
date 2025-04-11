import streamlit as st
import numpy as np
from scipy.stats import nbinom, poisson

# 模擬參數
MLB_STD_DEV = 3.8
NB_R = 5.3
NUM_SIMULATIONS = 50000

st.set_page_config(page_title="MLB 總分分析", layout="centered")
st.title("⚾ MLB 星級預測系統（含OBP進階模型）")

st.markdown("模擬三種分布預測總分過盤機率，並給出星級推薦")

st.subheader("主隊數據")
home = {
    'team_name': st.text_input("主隊名稱", "主隊"),
    'time_avg': st.number_input("時段場均得分", min_value=0.0),
    'base_avg': st.number_input("整體場均得分", min_value=0.0),
    'allow': st.number_input("主隊場均失分", min_value=0.0),
    'over_rate': st.slider("大分過盤率 (%)", 0, 100) / 100,
    'team_batting': st.number_input("團隊打擊率", min_value=0.000, max_value=1.000, value=0.250),
    'team_obp': st.number_input("團隊上壘率", min_value=0.000, max_value=1.000, value=0.320),
    'pitcher': {
        'era': st.number_input("主隊先發投手 ERA", min_value=0.0),
        'baa': st.number_input("主隊先發投手 BAA", min_value=0.000, max_value=1.000, value=0.250)
    }
}

st.subheader("客隊數據")
away = {
    'team_name': st.text_input("客隊名稱", "客隊"),
    'time_avg': st.number_input("時段場均得分（客）", min_value=0.0),
    'base_avg': st.number_input("整體場均得分（客）", min_value=0.0),
    'allow': st.number_input("客隊場均失分", min_value=0.0),
    'over_rate': st.slider("大分過盤率（客） (%)", 0, 100, key="客隊") / 100,
    'team_batting': st.number_input("團隊打擊率（客）", min_value=0.000, max_value=1.000, value=0.250),
    'team_obp': st.number_input("團隊上壘率（客）", min_value=0.000, max_value=1.000, value=0.320),
    'pitcher': {
        'era': st.number_input("客隊先發投手 ERA", min_value=0.0),
        'baa': st.number_input("客隊先發投手 BAA", min_value=0.000, max_value=1.000, value=0.250)
    }
}

target = st.number_input("盤口總分", min_value=0.0)

def simulate_game(home, away, target):
    def weighted_score(data, opp_allow, opp_pitcher, opp_team_stats):
        pitcher_factor = (opp_pitcher['era'] * 0.7 + opp_pitcher['baa'] * 0.3) / 4.5
        batting_factor = (data['team_batting'] / 0.250) * 0.6 + (data['team_obp'] / 0.320) * 0.4
        return (
            (data['time_avg'] * 0.35 +
             data['base_avg'] * 0.25 +
             data['allow'] * 0.15) *
            (1 + data['over_rate'] * 0.05) *
            (1.2 - pitcher_factor * 0.4) *
            (0.9 + batting_factor * 0.2)
        )

    home_score = weighted_score(home, away['allow'], away['pitcher'], away)
    away_score = weighted_score(away, home['allow'], home['pitcher'], home)
    combined_avg = home_score + away_score

    mc_scores = np.random.normal(home_score, MLB_STD_DEV, NUM_SIMULATIONS) +                 np.random.normal(away_score, MLB_STD_DEV, NUM_SIMULATIONS)
    mc_prob = np.mean(mc_scores > target) * 100

    def generate_nb(mean):
        p = NB_R / (NB_R + mean)
        return nbinom.rvs(NB_R, p, size=NUM_SIMULATIONS)

    nb_prob = np.mean((generate_nb(home_score) + generate_nb(away_score)) > target) * 100

    def generate_poisson(mean):
        return poisson.rvs(mean, size=NUM_SIMULATIONS)

    poisson_prob = np.mean((generate_poisson(home_score) + generate_poisson(away_score)) > target) * 100

    final_prob = mc_prob * 0.5 + nb_prob * 0.3 + poisson_prob * 0.2

    mean_diff = (combined_avg - target) / 3
    over_consistency = (home['over_rate'] + away['over_rate'] - 1)
    volatility = (np.std([home['time_avg'], home['base_avg']]) + np.std([away['time_avg'], away['base_avg']])) / 2
    trend_strength = ((home['time_avg'] - home['base_avg']) + (away['time_avg'] - away['base_avg'])) / 5

    if final_prob >= 50:
        base = 3.0 + (final_prob - 50) / 12.5
        adj = 0.5 * mean_diff + 0.7 * over_consistency + 0.3 * np.log1p(volatility) + 0.4 * trend_strength
    else:
        base = 3.0 - (50 - final_prob) / 12.5
        adj = -0.6 * mean_diff - 0.5 * over_consistency - 0.4 * np.log1p(volatility) - 0.3 * trend_strength

    raw_score = base + adj
    stars = min(5, max(1, round(raw_score, 1)))

    return home_score, away_score, combined_avg, mc_prob, nb_prob, poisson_prob, final_prob, stars

if st.button("🔍 開始分析"):
    hs, as_, total, mc, nb, poi, final, stars = simulate_game(home, away, target)
    st.subheader("📊 模型分析結果")
    st.markdown(f"- 主隊得分預期：`{hs:.2f}`")
    st.markdown(f"- 客隊得分預期：`{as_:.2f}`")
    st.markdown(f"- 預期總分：`{total:.2f}`（盤口 {target}）")
    st.markdown(f"- 蒙地卡羅：`{mc:.1f}%`")
    st.markdown(f"- 負二項分布：`{nb:.1f}%`")
    st.markdown(f"- 泊松分布：`{poi:.1f}%`")
    st.markdown(f"🎯 綜合機率：`{final:.1f}%`")

    recommendation = "🔥 建議：大分" if final >= 55 else "❄️ 建議：小分" if final <= 45 else "⚖️ 五五波"
    st.markdown(f"⭐ 星級評價：{'★' * int(stars)}{'☆' * (5 - int(round(stars)))} ({stars}/5.0)")
    st.success(recommendation)
