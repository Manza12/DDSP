import time


def debug_level_2_number(debug_level):
    if debug_level == "TRAIN":
        return 0
    elif debug_level == "RUN":
        return 1
    elif debug_level == "INFO":
        return 2
    elif debug_level == "DEBUG":
        return 3
    else:
        return None

def get_time(debug_level, debug_status):
    if debug_level_2_number(debug_level) <= debug_level_2_number(debug_status):
        return time.time()
    else:
        return None


def print_time(message, debug_level, debug_status, time_start, digits):
    if debug_level_2_number(debug_level) <= debug_level_2_number(debug_status):
        time_stamp = time.time()
        print(message, string_time(round(time_stamp - time_start, digits)))
        return time_stamp
    else:
        return None


def string_time(time_spent):
    seconds = time_spent % 60
    minutes = round(time_spent // 60)
    hours = round(time_spent // 3600)

    if minutes == 0 and hours == 0:
        return str(seconds) + " s"
    elif hours == 0:
        return str(minutes) + " m " + str(seconds) + " s"
    else:
        return str(hours) + " h " + str(minutes) + " m " + str(seconds) + " s"


if __name__ == "__main__":
    DEBUG_LEVEL = "TRAIN"
    DEBUG_STATUS = "DEBUG"
    TIME_START = get_time(DEBUG_LEVEL, DEBUG_STATUS)
    time.sleep(1)
    time_end = print_time("Time spent :", DEBUG_LEVEL, DEBUG_STATUS, TIME_START, 6)
