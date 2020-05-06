import csv
import os
import sys

# Creates summary metrics for one individual

def calc_app_usage(eventlist, app):
    pairs = []

    start = None
    stop = None

    for event in eventlist:
        if event["app"] == app:
            if event["action"] == "start":
                start = event["timestamp"]
            elif event["action"] == "stop":
                stop = event["timestamp"]
                pairs.append((float(start), float(stop)))

    subtract = lambda x : x[1] - x[0]
    times = list(map(subtract, pairs))
    return sum(times), sum(times) / len(times), times

def summarize_task(dirname):
    os.chdir(dirname)
    print("\n------------------------------")
    print("Summarizing task: ", dirname)

    start_time = None
    end_time = None

    with open('app_usage.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)[4:]

        # Completion time
        start_time = float(rows[0]['timestamp'])
        end_time = float(rows[-1]['timestamp'])
        total_time = end_time - start_time
        
        # LL App Usage
        ll_usage, ll_avg, ll_times = calc_app_usage(rows, "Lang. Learn")
        ll_percent = ll_usage / total_time

        # TD App Usage
        td_usage, td_avg, td_times = calc_app_usage(rows, "TODO")
        td_percent = td_usage / total_time

        # WK App Usage
        wk_usage, wk_avg, wk_times= calc_app_usage(rows, "Packer")
        wk_percent = wk_usage / total_time

        session_times = ll_times + td_times + wk_times
        session_avg = sum(session_times) / len(session_times)

    # Object Detection Time
    with open('object_detections.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        detections_time = float(rows[-1]["timestamp"]) - start_time

    # App Switch Behaviour
    with open('app_switch.csv', 'r') as f:
        reader = csv.DictReader(f, fieldnames=["event", "timestamp", "from_app", "to_app", "action"])
        rows = list(reader)[2:]
        
        total_switches = len(rows)
        ll_cases = [r for r in rows if r["to_app"] == "Lang. Learn"]
        ll_switches = len(ll_cases)

        td_cases = [r for r in rows if r["to_app"] == "TODO"]
        td_switches = len(td_cases)

        wk_cases = [r for r in rows if r["to_app"] == "Packer"]
        wk_switches = len(wk_cases)

        manual_cases = [r for r in rows if r["action"] == "manual"]
        manual_switches = len(manual_cases)
        insitu_cases = [r for r in rows if r["action"] == "contextMenu"]
        insitu_switches = len(insitu_cases)

    # Print summary
    print("Total Time: {:.2f} s".format(total_time))
    print("Objects Time: {:.2f} s".format(detections_time))


    print("\nAvg app session: {:.2f} s".format(session_avg))
    print("LL Usage: {:.2f} s,  {:.2%}".format(ll_usage, ll_percent))
    print("LL Avg: {:.2f} s".format(ll_avg))

    print("TD Usage: {:.2f} s,  {:.2%}".format(td_usage, td_percent))
    print("TD Avg: {:.2f} s".format(td_avg))

    print("WK Usage: {:.2f} s,  {:.2%}".format(wk_usage, wk_percent))
    print("WK Avg: {:.2f} s".format(wk_avg))

    print("\nTotal Switches: {}".format(total_switches))
    print("Per app switches:  LL {}, TD {}, WK {}".format(ll_switches, td_switches, wk_switches))
    print("Per type switches: Manual {} Insitu {}".format(manual_switches, insitu_switches))

    os.chdir("..")

def summarize(dirname):
    os.chdir(dirname)
    summarize_task('manual')
    summarize_task('insitu')
    summarize_task('manual-insitu')

if __name__ == '__main__':
    summarize(sys.argv[1])