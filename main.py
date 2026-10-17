import os, json, sqlite3, datetime as dt, time
from contextlib import closing
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI, APITimeoutError

DB = os.getenv("DB_PATH", "data.db")
TOKEN = os.getenv("APP_TOKEN", "")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")  # V4.1-Flash, 支持图片
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY", ""), base_url="https://api.deepseek.com", max_retries=0)


def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


with closing(db()) as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS profile(id INTEGER PRIMARY KEY CHECK(id=1), sex TEXT, age INT,
      height_cm REAL, weight_kg REAL, target_weight_kg REAL, target_date TEXT, bmr REAL);
    CREATE TABLE IF NOT EXISTS meals(id INTEGER PRIMARY KEY, date TEXT, time TEXT, meal_type TEXT,
      name TEXT, amount TEXT, kcal REAL);
    CREATE TABLE IF NOT EXISTS workouts(id INTEGER PRIMARY KEY, date TEXT, name TEXT, detail TEXT, kcal REAL);
    CREATE TABLE IF NOT EXISTS weights(id INTEGER PRIMARY KEY, date TEXT, kg REAL);
    CREATE TABLE IF NOT EXISTS sleeps(id INTEGER PRIMARY KEY, date TEXT, sleep_time TEXT, wake_time TEXT,
      nap_sleep_time TEXT, nap_wake_time TEXT, note TEXT);
    CREATE INDEX IF NOT EXISTS i_wt ON weights(date); CREATE INDEX IF NOT EXISTS i_sl ON sleeps(date);
    CREATE INDEX IF NOT EXISTS i_m ON meals(date); CREATE INDEX IF NOT EXISTS i_w ON workouts(date);
    """)
    cols = {r["name"] for r in c.execute("PRAGMA table_info(sleeps)")}
    for col in ("nap_sleep_time", "nap_wake_time"):  # 兼容旧库
        if col not in cols:
            c.execute(f"ALTER TABLE sleeps ADD COLUMN {col} TEXT")
    pcols = {r["name"] for r in c.execute("PRAGMA table_info(profile)")}
    if "bmr" not in pcols:  # 兼容旧库
        c.execute("ALTER TABLE profile ADD COLUMN bmr REAL")
    c.commit()


def auth(x_token: str = Header("")):
    if TOKEN and x_token != TOKEN:
        raise HTTPException(401, "bad token")


app = FastAPI(dependencies=[Depends(auth)])


def get_profile():
    with closing(db()) as c:
        r = c.execute("SELECT * FROM profile WHERE id=1").fetchone()
    return dict(r) if r else None


class Profile(BaseModel):
    sex: str = "male"
    age: int
    height_cm: float
    weight_kg: float
    target_weight_kg: float
    target_date: str
    bmr: float | None = None


@app.get("/api/profile")
def profile_get():
    return get_profile() or {}


@app.post("/api/profile")
def profile_set(p: Profile):
    try:
        ok = dt.date.fromisoformat(p.target_date) > dt.date.today()
    except ValueError:
        ok = False
    if not ok:
        raise HTTPException(400, "目标日期必须晚于今天")
    with closing(db()) as c:
        c.execute("REPLACE INTO profile VALUES(1,?,?,?,?,?,?,?)",
                  (p.sex, p.age, p.height_cm, p.weight_kg, p.target_weight_kg, p.target_date, p.bmr))
        c.commit()
    return {"ok": True}


def bmr_value(p=None):
    """基础代谢率：优先用档案里手填的 bmr，否则按 Mifflin-St Jeor 公式估算（体重优先取最近一次称重）"""
    p = p or get_profile()
    if not p:
        return 0
    with closing(db()) as c:
        r = c.execute("SELECT kg FROM weights ORDER BY date DESC, id DESC LIMIT 1").fetchone()
    weight = r["kg"] if r else p["weight_kg"]  # 优先用最近一次称重
    return p["bmr"] or (10 * weight + 6.25 * p["height_cm"] - 5 * p["age"] + (5 if p["sex"] == "male" else -161))


def budget(burn, p=None):
    """基础代谢*1.2(久坐系数,运动单独记) - 为达成目标每日需要的热量缺口 + 运动消耗"""
    p = p or get_profile()
    if not p:
        return {}
    with closing(db()) as c:
        r = c.execute("SELECT kg FROM weights ORDER BY date DESC, id DESC LIMIT 1").fetchone()
    weight = r["kg"] if r else p["weight_kg"]  # 优先用最近一次称重
    bmr = bmr_value(p)
    base = bmr * 1.2
    try:
        days = max((dt.date.fromisoformat(p["target_date"]) - dt.date.today()).days, 1)
    except Exception:
        days = 90
    deficit = min(max((weight - p["target_weight_kg"]) * 7700 / days, 0), 1000)
    return {"bmr": round(bmr), "base": round(base), "deficit": round(deficit),
            "budget": round(base - deficit + burn)}


@app.get("/api/day")
def day(date: str):
    with closing(db()) as c:
        meals = [dict(r) for r in c.execute("SELECT * FROM meals WHERE date=? ORDER BY time,id", (date,))]
        works = [dict(r) for r in c.execute("SELECT * FROM workouts WHERE date=? ORDER BY id", (date,))]
        weights = [dict(r) for r in c.execute("SELECT * FROM weights WHERE date=? ORDER BY id", (date,))]
        sleeps = [dict(r) for r in c.execute("SELECT * FROM sleeps WHERE date=? ORDER BY id", (date,))]
    intake = sum(m["kcal"] or 0 for m in meals)
    burn = sum(w["kcal"] or 0 for w in works)
    return {"meals": meals, "workouts": works, "weights": weights, "sleeps": sleeps, "intake": round(intake), "burn": round(burn), **budget(burn)}


def sleep_minutes(sleep_time, wake_time):
    """计算睡眠时长（分钟）。支持完整日期时间（YYYY-MM-DD HH:MM）或纯时间（HH:MM，跨零点自动加一天）；缺任一时间返回 0"""
    if not sleep_time or not wake_time:
        return 0
    try:
        a = dt.datetime.fromisoformat(sleep_time)
        b = dt.datetime.fromisoformat(wake_time)
    except ValueError:
        try:
            a = dt.datetime.strptime(sleep_time, "%H:%M")
            b = dt.datetime.strptime(wake_time, "%H:%M")
        except ValueError:
            return 0
        m = int((b - a).total_seconds() // 60)
        return m + 1440 if m < 0 else m
    m = int((b - a).total_seconds() // 60)
    return m if m > 0 else 0


def row_sleep_minutes(s):
    """一条睡眠记录的总时长：夜间睡眠 + 午休"""
    return sleep_minutes(s["sleep_time"], s["wake_time"]) + sleep_minutes(s["nap_sleep_time"], s["nap_wake_time"])


@app.get("/api/month")
def month(month: str):  # YYYY-MM
    like = month + "%"
    with closing(db()) as c:
        rows = c.execute("""SELECT date, SUM(i) i, SUM(b) b FROM (
            SELECT date, kcal i, 0 b FROM meals WHERE date LIKE ? UNION ALL
            SELECT date, 0, kcal FROM workouts WHERE date LIKE ?) GROUP BY date""", (like, like)).fetchall()
        ws = c.execute("SELECT date, kg FROM weights WHERE date LIKE ? ORDER BY id", (like,)).fetchall()
        ss = c.execute("SELECT date, sleep_time, wake_time, nap_sleep_time, nap_wake_time FROM sleeps WHERE date LIKE ? ORDER BY id", (like,)).fetchall()
    out = {r["date"]: {"intake": round(r["i"] or 0), "burn": round(r["b"] or 0), "weight": None, "sleep": None} for r in rows}
    for w in ws:  # 同一天多次称重取最后一次
        out.setdefault(w["date"], {"intake": 0, "burn": 0, "weight": None, "sleep": None})["weight"] = w["kg"]
    for s in ss:  # 同一天多条睡眠累加时长（分钟）
        m = row_sleep_minutes(s)
        if m:
            d = out.setdefault(s["date"], {"intake": 0, "burn": 0, "weight": None, "sleep": None})
            d["sleep"] = (d["sleep"] or 0) + m
    base = round(bmr_value() * 1.2)  # 基础代谢*1.2(久坐系数)，与记录页预算口径一致
    for v in out.values():
        v.setdefault("weight", None)
        v.setdefault("sleep", None)
        v["base"] = base
    return out


@app.get("/api/weeks")
def weeks(month: str):  # YYYY-MM，返回该月覆盖到的各周（周一至周日）平均体重
    y, m = int(month[:4]), int(month[5:7])
    first = dt.date(y, m, 1)
    last = dt.date(y, m, 1) + dt.timedelta(days=31)
    last = last.replace(day=1) - dt.timedelta(days=1)
    start = first - dt.timedelta(days=first.weekday())  # 回退到周一
    end = last + dt.timedelta(days=6 - last.weekday())  # 前进到周日
    with closing(db()) as c:
        rows = c.execute("SELECT date, kg FROM weights WHERE date>=? AND date<=? ORDER BY date, id",
                         (start.isoformat(), end.isoformat())).fetchall()
    by_day = {}  # 同一天多次称重取最后一次
    for r in rows:
        by_day[r["date"]] = r["kg"]
    out = []
    d = start
    while d <= end:
        vals = [by_day[k] for k in ((d + dt.timedelta(days=i)).isoformat() for i in range(7)) if k in by_day]
        if vals:
            out.append({"start": d.isoformat(), "end": (d + dt.timedelta(days=6)).isoformat(),
                        "avg": round(sum(vals) / len(vals), 2), "days": len(vals)})
        d += dt.timedelta(days=7)
    return out


@app.delete("/api/{kind}/{id}")
def delete(kind: str, id: int):
    if kind not in ("meals", "workouts", "weights", "sleeps"):
        raise HTTPException(404)
    with closing(db()) as c:
        c.execute(f"DELETE FROM {kind} WHERE id=?", (id,))
        c.commit()
    return {"ok": True}


SYSTEM = """你是饮食与运动记录助手。把用户消息拆分成结构化记录，只输出一个JSON对象，不要任何其他文字：
{{"date":"YYYY-MM-DD","meals":[{{"time":"HH:MM","meal_type":"早餐|午餐|晚餐|加餐","name":"","amount":"","kcal":0}}],
"workouts":[{{"name":"","detail":"","kcal":0}}],"weights":[{{"kg":0.0}}],"sleeps":[{{"sleep_time":"YYYY-MM-DD HH:MM","wake_time":"YYYY-MM-DD HH:MM","nap_sleep_time":"YYYY-MM-DD HH:MM","nap_wake_time":"YYYY-MM-DD HH:MM","note":""}}],"reply":"一句简短中文说明/估算依据，需要追问时写在这里"}}
规则：
1. 今天是{today}，现在时间{now}，用户当前查看的日期是{date}。date默认取查看日期；消息里若有明确日期（如“1001”表示10月1日，年份取今年）则用该日期。
2. 用户直接给出kcal就原样使用；否则根据食物和份量估算（拳头大小≈一个中等水果/约150-200g主食），kcal为整数。
3. 时间：用户说“刚起床/刚吃/现在”等用现在时间；说了具体时间用该时间；否则早餐08:00、午餐12:30、晚餐18:30、加餐16:00。meal_type按用户描述或时间判断。
4. 运动：每个动作/项目单独一条，name为项目名，detail保留用户原始数据（时长/组数x次数/重量），kcal按体重{weight}kg估算该项目消耗；力量训练考虑组数、次数、重量和间歇；辅助引体向上的重量是辅助重量。
5. 用户发的图片如果是食物就识别并估算；如果是营养成分表就按表计算。
6. 体重：用户提到称重/体重数值时记入weights（kg，一天可多次），不要估算体重。
7. 作息：一条sleeps代表一天的睡眠，包含夜间睡眠和午休两部分，时间一律用完整日期时间“YYYY-MM-DD HH:MM”。
   - 夜间睡眠：sleep_time是“昨晚睡觉时间”，日期取查看日期的前一天；wake_time是“今天起床时间”，日期取查看日期。例如查看日期是{date}，“昨晚11点半睡、今早7点起”=sleep_time为前一天23:30、wake_time为{date} 07:00。
   - 午休：nap_sleep_time是午休入睡时间、nap_wake_time是午休起床时间，日期都取查看日期（如“中午睡了半小时”=nap_sleep_time为{date} 13:00、nap_wake_time为{date} 13:30）。没有午休时两个字段填空字符串。
   - 只知夜间或午休其一时，另一个填空字符串；note只写用户提到的睡眠情况。作息不影响热量计算；同一条消息可同时包含饮食、运动、体重、作息。
8. 无法判断时所有数组留空，并在reply里提问。
9. 引用历史：用户说“和昨天/前天/某天吃的一样”“照旧”“同上”等时，从下方【历史饮食记录】中找到对应日期的饮食，原样复制其name、amount、kcal（meal_type、time可沿用原记录），date用用户当前要记录的日期（默认查看日期）。历史里没有对应日期时不要编造，在reply里说明并追问。
10. 多轮对话：本次请求可能包含同一会话中此前的对话消息（user/assistant）。当用户当前消息是补充、修正或指代上文时（如“再加一个鸡蛋”“刚才那个改成200g”“也是晚餐”），结合上文理解并只输出本次要新增/修正的记录；不要重复输出上文已确认过的记录。
【历史饮食记录】
{history}"""


class ParseIn(BaseModel):
    text: str = ""
    images: list[str] = []  # data URI
    date: str
    today: str
    now: str
    history: list[dict] = []  # 本会话已发生的多轮对话：[{"role":"user|assistant","content":"..."}]


def meal_history(days: int = 7) -> str:
    """最近若干天的饮食记录，供模型处理“和昨天吃的一样”等引用"""
    with closing(db()) as c:
        rows = c.execute("SELECT date,time,meal_type,name,amount,kcal FROM meals ORDER BY date DESC,time,id").fetchall()
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r)
    lines = []
    for d in sorted(by_date, reverse=True)[:days]:
        items = "；".join(f'{r["meal_type"]} {r["name"]} {r["amount"] or ""} {round(r["kcal"] or 0)}kcal'.strip()
                          for r in by_date[d])
        lines.append(f"{d}: {items}")
    return "\n".join(lines) if lines else "（暂无历史记录）"


# 引用历史饮食的触发词：命中才查库，避免每次请求都读 meals 表
HISTORY_HINTS = ("昨天", "前天", "昨天吃", "一样", "照旧", "同上", "老样子", "老规矩", "和上次", "跟上次", "之前吃")


def needs_history(text: str) -> bool:
    """判断用户消息是否在引用历史饮食（如“和昨天一样”“照旧”）"""
    return any(k in text for k in HISTORY_HINTS)


def call_model(msgs):
    """调用模型；请求超过 10s 失败（多为冷连接/网络抖动）则重试，最多重试 2 次后放弃"""
    for i in range(3):  # 首次 + 最多 2 次重试
        t = time.time()
        try:
            return client.chat.completions.create(
                model=MODEL, response_format={"type": "json_object"}, timeout=60, messages=msgs)
        except Exception:
            if time.time() - t < 10 or i == 2:  # 快速失败（如参数错误）不重试；已重试 2 次则放弃
                raise


@app.post("/api/parse")
def parse(b: ParseIn):
    p = get_profile() or {}
    sys = SYSTEM.format(today=b.today, now=b.now, date=b.date, weight=p.get("weight_kg", 65),
                        history=meal_history() if needs_history(b.text) else "（本次未引用历史记录）")
    content = [{"type": "text", "text": b.text or "（见图片）"}]
    content += [{"type": "image_url", "image_url": {"url": u}} for u in b.images]
    msgs = [{"role": "system", "content": sys}]
    for h in b.history[-10:]:  # 仅保留最近若干轮，控制上下文长度
        role = h.get("role")
        if role in ("user", "assistant") and h.get("content"):
            msgs.append({"role": role, "content": h["content"]})
    msgs.append({"role": "user", "content": content})
    try:
        r = call_model(msgs)
        data = json.loads(r.choices[0].message.content)
    except APITimeoutError:
        raise HTTPException(504, "模型响应超时（60s），请重试")
    except Exception as e:
        raise HTTPException(502, f"模型调用/解析失败: {e}")
    data.setdefault("date", b.date)
    data.setdefault("meals", [])
    data.setdefault("workouts", [])
    data.setdefault("weights", [])
    data.setdefault("sleeps", [])
    return data


ADVICE_SYSTEM = """你是营养师。根据用户某天的饮食记录，评估营养是否均衡、健康、是否利于减脂，只输出一个JSON对象，不要任何其他文字：
{{"macros":{{"carb":0,"protein":0,"fat":0}},"score":0,"verdict":"","tips":["",""]}}
字段说明：
- macros：估算当天碳水、蛋白质、脂肪的克数（整数）。
- score：0-100 的饮食质量分，综合营养均衡度、食物质量、减脂适配度。
- verdict：一句话总评（不超过30字），点明是否均衡、是否利于减脂。
- tips：2-3条具体可执行的改进建议，每条不超过25字，针对当天缺口给出食物建议。
规则：
1. 用户资料：性别{sex}，年龄{age}岁，身高{height}cm，体重{weight}kg，目标体重{target}kg，每日热量预算{budget}kcal，当天摄入{intake}kcal，运动消耗{burn}kcal。
2. 参考中国居民膳食指南与减脂期常见配比（蛋白质约每公斤体重1.2-1.6g，脂肪占总热量20-30%，碳水占45-55%），判断三大营养素是否达标与比例是否合理。
3. 结合食物种类判断质量：蔬果、全谷物、优质蛋白是否充足，精制碳水、油炸、含糖饮料、加工肉是否偏多。
4. 前一天饮食仅作趋势参考：若同类问题连续出现，建议可更明确；若只是单日波动，不必过度提醒。
5. 语气简洁务实，不堆砌术语，不输出markdown。
【当天饮食】
{today}
【前一天饮食】
{yesterday}"""


WORKOUT_SYSTEM = """你是运动教练。根据用户近7天的运动与作息记录，评估当天运动是否合理、是否利于减脂，只输出一个JSON对象，不要任何其他文字：
{{"score":0,"verdict":"","volume":"","tips":["",""]}}
字段说明：
- score：0-100 的运动质量分，综合运动量、强度、频率、恢复情况。
- verdict：一句话总评（不超过30字），点明当天运动是否合理、是否利于减脂。
- volume：当天运动量概述（不超过20字），如“力量40分钟+有氧20分钟，消耗约350kcal”。
- tips：2-3条具体可执行的建议，每条不超过25字，针对训练安排或恢复给出。
规则：
1. 用户资料：性别{sex}，年龄{age}岁，体重{weight}kg，目标体重{target}kg，每日热量预算{budget}kcal。
2. 参考减脂期运动建议：每周3-5次运动，力量训练与有氧结合，单次有效运动30-60分钟，避免连续高强度无休息。
3. 结合近7天记录判断频率与恢复：若连续多天高强度训练，提醒休息；若运动频率过低，给出可执行的起步建议。
4. 作息影响恢复：睡眠不足时降低强度建议，并提示睡眠对减脂的作用。
5. 语气简洁务实，不堆砌术语，不输出markdown。
【当天运动】
{today}
【近7天运动】
{week}
【近7天作息】
{sleep}"""


class AdviceIn(BaseModel):
    date: str


def day_meals_text(date: str) -> str:
    """把某天的饮食记录拼成一行文本，供营养分析使用"""
    with closing(db()) as c:
        rows = c.execute("SELECT time,meal_type,name,amount,kcal FROM meals WHERE date=? ORDER BY time,id",
                         (date,)).fetchall()
    if not rows:
        return "（无记录）"
    return "；".join(f'{r["meal_type"]} {r["name"]} {r["amount"] or ""} {round(r["kcal"] or 0)}kcal'.replace("  ", " ").strip()
                    for r in rows)


def day_workouts_text(date: str) -> str:
    """把某天的运动记录拼成一行文本"""
    with closing(db()) as c:
        rows = c.execute("SELECT name,detail,kcal FROM workouts WHERE date=? ORDER BY id", (date,)).fetchall()
    if not rows:
        return "（无记录）"
    return "；".join(f'{r["name"]} {r["detail"] or ""} {round(r["kcal"] or 0)}kcal'.replace("  ", " ").strip()
                    for r in rows)


def recent_workouts_text(date: str, days: int = 7) -> str:
    """近若干天的运动汇总，按日期分组"""
    start = (dt.date.fromisoformat(date) - dt.timedelta(days=days - 1)).isoformat()
    with closing(db()) as c:
        rows = c.execute("SELECT date,name,detail,kcal FROM workouts WHERE date>=? AND date<=? ORDER BY date,id",
                         (start, date)).fetchall()
    if not rows:
        return "（无记录）"
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r)
    return "\n".join(f'{d}: ' + "；".join(f'{r["name"]} {r["detail"] or ""} {round(r["kcal"] or 0)}kcal'.replace("  ", " ").strip()
                                          for r in by_date[d])
                     for d in sorted(by_date))


def recent_sleep_text(date: str, days: int = 7) -> str:
    """近若干天的睡眠时长汇总"""
    start = (dt.date.fromisoformat(date) - dt.timedelta(days=days - 1)).isoformat()
    with closing(db()) as c:
        rows = c.execute("SELECT date,sleep_time,wake_time,nap_sleep_time,nap_wake_time FROM sleeps "
                         "WHERE date>=? AND date<=? ORDER BY date,id", (start, date)).fetchall()
    by_date = {}
    for r in rows:
        by_date[r["date"]] = by_date.get(r["date"], 0) + row_sleep_minutes(r)
    if not by_date:
        return "（无记录）"
    return "\n".join(f'{d}: {m // 60}h{m % 60}m' for d, m in sorted(by_date.items()) if m)


@app.post("/api/advice")
def advice(b: AdviceIn):
    p = get_profile() or {}
    with closing(db()) as c:
        intake = c.execute("SELECT COALESCE(SUM(kcal),0) FROM meals WHERE date=?", (b.date,)).fetchone()[0]
        burn = c.execute("SELECT COALESCE(SUM(kcal),0) FROM workouts WHERE date=?", (b.date,)).fetchone()[0]
    if not intake:
        raise HTTPException(400, "当天还没有饮食记录")
    prev = (dt.date.fromisoformat(b.date) - dt.timedelta(days=1)).isoformat()
    sys = ADVICE_SYSTEM.format(
        sex="女" if p.get("sex") == "female" else "男", age=p.get("age", "-"), height=p.get("height_cm", "-"),
        weight=p.get("weight_kg", "-"), target=p.get("target_weight_kg", "-"),
        budget=budget(burn).get("budget", "-"), intake=round(intake), burn=round(burn),
        today=day_meals_text(b.date), yesterday=day_meals_text(prev))
    try:
        r = call_model([{"role": "system", "content": sys},
                        {"role": "user", "content": f"请评估 {b.date} 的饮食"}])
        data = json.loads(r.choices[0].message.content)
    except APITimeoutError:
        raise HTTPException(504, "模型响应超时（60s），请重试")
    except Exception as e:
        raise HTTPException(502, f"模型调用/解析失败: {e}")
    data.setdefault("macros", {})
    data.setdefault("score", 0)
    data.setdefault("verdict", "")
    data.setdefault("tips", [])
    return data


@app.post("/api/workout-advice")
def workout_advice(b: AdviceIn):
    p = get_profile() or {}
    with closing(db()) as c:
        burn = c.execute("SELECT COALESCE(SUM(kcal),0) FROM workouts WHERE date=?", (b.date,)).fetchone()[0]
    if not burn:
        raise HTTPException(400, "当天还没有运动记录")
    sys = WORKOUT_SYSTEM.format(
        sex="女" if p.get("sex") == "female" else "男", age=p.get("age", "-"),
        weight=p.get("weight_kg", "-"), target=p.get("target_weight_kg", "-"),
        budget=budget(burn).get("budget", "-"),
        today=day_workouts_text(b.date), week=recent_workouts_text(b.date), sleep=recent_sleep_text(b.date))
    try:
        r = call_model([{"role": "system", "content": sys},
                        {"role": "user", "content": f"请评估 {b.date} 的运动"}])
        data = json.loads(r.choices[0].message.content)
    except APITimeoutError:
        raise HTTPException(504, "模型响应超时（60s），请重试")
    except Exception as e:
        raise HTTPException(502, f"模型调用/解析失败: {e}")
    data.setdefault("score", 0)
    data.setdefault("verdict", "")
    data.setdefault("volume", "")
    data.setdefault("tips", [])
    return data


class SummaryIn(BaseModel):
    date: str
    diet: dict = {}  # 当天饮食建议结果
    workout: dict = {}  # 当天运动建议结果


SUMMARY_SYSTEM = """你是减脂顾问。根据用户当天的饮食建议与运动建议，给出综合结论，只输出一个JSON对象，不要任何其他文字：
{{"score":0,"verdict":"","focus":"","tips":["",""]}}
字段说明：
- score：0-100 的综合分，综合饮食与运动两方面。
- verdict：一句话总评（不超过30字），点明当天整体是否利于减脂。
- focus：当天最需要改进的一件事（不超过20字）。
- tips：2-3条综合建议，每条不超过25字，兼顾饮食与运动，避免与已有建议重复。
规则：
1. 用户资料：性别{sex}，年龄{age}岁，体重{weight}kg，目标体重{target}kg，每日热量预算{budget}kcal，当天摄入{intake}kcal，运动消耗{burn}kcal。
2. 综合饮食与运动两方面判断：热量缺口是否合理、蛋白质是否充足、运动量是否匹配、恢复是否足够。
3. 若饮食与运动建议存在冲突或互补，明确指出；若某方面数据缺失，说明局限并给出可执行建议。
4. 语气简洁务实，不堆砌术语，不输出markdown。
【当天饮食建议】
{diet}
【当天运动建议】
{workout}"""


@app.post("/api/summary")
def summary(b: SummaryIn):
    p = get_profile() or {}
    with closing(db()) as c:
        intake = c.execute("SELECT COALESCE(SUM(kcal),0) FROM meals WHERE date=?", (b.date,)).fetchone()[0]
        burn = c.execute("SELECT COALESCE(SUM(kcal),0) FROM workouts WHERE date=?", (b.date,)).fetchone()[0]
    if not b.diet and not b.workout:
        raise HTTPException(400, "请先生成饮食建议或运动建议")
    sys = SUMMARY_SYSTEM.format(
        sex="女" if p.get("sex") == "female" else "男", age=p.get("age", "-"),
        weight=p.get("weight_kg", "-"), target=p.get("target_weight_kg", "-"),
        budget=budget(burn).get("budget", "-"), intake=round(intake), burn=round(burn),
        diet=json.dumps(b.diet, ensure_ascii=False) if b.diet else "（未生成）",
        workout=json.dumps(b.workout, ensure_ascii=False) if b.workout else "（未生成）")
    try:
        r = call_model([{"role": "system", "content": sys},
                        {"role": "user", "content": f"请给出 {b.date} 的综合建议"}])
        data = json.loads(r.choices[0].message.content)
    except APITimeoutError:
        raise HTTPException(504, "模型响应超时（60s），请重试")
    except Exception as e:
        raise HTTPException(502, f"模型调用/解析失败: {e}")
    data.setdefault("score", 0)
    data.setdefault("verdict", "")
    data.setdefault("focus", "")
    data.setdefault("tips", [])
    return data


class Commit(BaseModel):
    date: str
    meals: list[dict] = []
    workouts: list[dict] = []
    weights: list[dict] = []
    sleeps: list[dict] = []


@app.post("/api/commit")
def commit(b: Commit):
    with closing(db()) as c:
        for m in b.meals:
            c.execute("INSERT INTO meals(date,time,meal_type,name,amount,kcal) VALUES(?,?,?,?,?,?)",
                      (b.date, m.get("time"), m.get("meal_type"), m.get("name"), m.get("amount"), m.get("kcal", 0)))
        for w in b.workouts:
            c.execute("INSERT INTO workouts(date,name,detail,kcal) VALUES(?,?,?,?)",
                      (b.date, w.get("name"), w.get("detail"), w.get("kcal", 0)))
        for w in b.weights:
            if w.get("kg"):
                c.execute("INSERT INTO weights(date,kg) VALUES(?,?)", (b.date, w["kg"]))
        for x in b.sleeps:
            c.execute("INSERT INTO sleeps(date,sleep_time,wake_time,nap_sleep_time,nap_wake_time,note) VALUES(?,?,?,?,?,?)",
                      (b.date, x.get("sleep_time"), x.get("wake_time"),
                       x.get("nap_sleep_time"), x.get("nap_wake_time"), x.get("note")))
        c.commit()
    return {"ok": True}


@app.middleware("http")
async def cache_static(request, call_next):
    """静态资源加缓存头，减少公网重复下载；API 响应不缓存；index.html 每次校验，保证发版即时生效"""
    r = await call_next(request)
    path = request.url.path
    if not path.startswith("/api/"):
        if path in ("/", "/index.html"):
            r.headers["Cache-Control"] = "no-cache"
        else:
            r.headers["Cache-Control"] = "public, max-age=300, must-revalidate"
    return r


app.mount("/", StaticFiles(directory="static", html=True))
