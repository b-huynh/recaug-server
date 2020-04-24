import csv
import os
import sys

# Interaction stats
app_usage_out = "app_usage.csv"
app_usage = ["event,timestamp,app,action\n"]

app_switch_out = "app_switch.csv"
app_switch = ["event,timestamp,app,action\n"]

menu_out = "menu.csv"
menu = ["event,timestamp,action\n"]

context_menu_out = "context_menu.csv"
context_menu = ["event,timestamp,app,action\n"]

hover_out = "hover.csv"
hover = ["event,timestamp,app,action\n"]

# Object recognition stats
detections_out = "object_detections.csv"
detections = ["event,timestamp,object,position\n"]

# Task stats
activities_out = "activites.csv"
activities = ["event,timestamp,object,app,response\n"]

def trim_row(r):
    cols = r.split(',')
    for i in range(len(cols)):
        trimmed = cols[i].lstrip().rstrip()
        cols[i] = trimmed

    if r.startswith("activity"):
        response = cols[4]
        if response == "- Remove":
            cols[4] = "Remove"
        if response == "+ Create Ticket":
            cols[4] = "Create Ticket"
        if response == "+":
            cols[4] = "Plus"
        if response == "✓":
            cols[4] = "Check"

    return ','.join(cols) + '\n'

def tocsv(fname):
    # Parse activity log
    with open(fname, "r", encoding="utf-8") as in_f:
        for row in in_f:
            if row.startswith("start"):
                start = float(row.split(',')[1])
            if row.startswith("complete"):
                complete_str = row.split(',')[1]
                complete = float(row.split(',')[1])

            if row.startswith("appUsage"):
                app_usage.append(trim_row(row))
            if row.startswith("appSwitch"):
                app_switch.append(trim_row(row))
            if row.startswith("menu"):
                menu.append(trim_row(row))
            if row.startswith("contextMenu"):
                context_menu.append(trim_row(row))
            if row.startswith("hover"):
                hover.append(trim_row(row))
            
            if row.startswith("object"):
                detections.append(trim_row(row))
            if row.startswith("activity"):
                activities.append(trim_row(row))

    # Create new directory
    new_dir = os.path.splitext(fname)[0]
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    os.chdir(new_dir)

    # Write 
    with open(app_usage_out, "w") as f:
        f.writelines(app_usage)

        # Add final app stop row which is same time as completion time
        last_app_cols = app_usage[-1].split(',')
        last_app_cols[1] = complete_str
        last_app_cols[3] = "stop"
        app_stop_row = ",".join(last_app_cols)
        f.write(app_stop_row)

    with open(app_switch_out, "w") as f:
        f.writelines(app_switch)
    with open(menu_out, "w") as f:
        f.writelines(menu)
    with open(context_menu_out, "w") as f:
        f.writelines(context_menu)
    with open(hover_out, "w") as f:
        f.writelines(hover)
    with open(detections_out, "w") as f:
        f.writelines(detections)        
    with open(activities_out, "w") as f:
        f.writelines(activities)

    def get_timestamp(r):
        return float(r.split(',')[1])

    time_complete = complete - start
    print("Completion time: ", time_complete)
    with open("summary.txt", "w") as f:
        f.write("Start: {0} seconds\n".format(start))
        f.write("Complete: {0} seconds\n".format(complete))
        f.write("Total: {0} seconds\n".format(time_complete))

        detections_complete = get_timestamp(detections[-1]) - start
        f.write("Detections Complete: {0} seconds\n".format(detections_complete))

def main():
    fname = sys.argv[1]
    tocsv(fname)

if __name__ == '__main__':
    main()