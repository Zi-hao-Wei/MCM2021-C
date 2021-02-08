import datetime 
import pandas as pd
from pyecharts import options as opts 
from pyecharts.charts import Calendar

begin = pd.datetime(2021, 1, 1)
count, data = 0, []

val=0
update={59:(200,40),139:(200,80),159:(200,40),179:(200,80),189:(200,40),199:(200,80),219:(200,40),239:(200,80),269:(200,40),299:(200,80),329:(200,0)}

for _ in range(365):
    if (count in update):
        data.append([str(begin + datetime.timedelta(count)), update[count][0]])
        val=update[count][1]+1
        print(count,val)
    else:
        print(count,val)
        data.append([str(begin + datetime.timedelta(count)), val])
    count += 1
c = (
    Calendar()
    .add("", data, calendar_opts=opts.CalendarOpts(range_="2021"))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="2021"),
        visualmap_opts=opts.VisualMapOpts(
            max_=200,
            min_=0,
            orient="horizontal",
            is_piecewise=True,
            pos_top="230px",
            pos_left="100px",
        ),
    )
)
c.render("Calender.html")