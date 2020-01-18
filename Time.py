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
        return 100


def print_time(message, debug_level, debug_status, time_start, digits):
    time_stamp = time.time()
    print_info(message + " " + string_time(time_stamp - time_start, digits),
               debug_level, debug_status)
    return time_stamp


def print_info(message, debug_level, debug_status):
    if debug_level_2_number(debug_level) >= debug_level_2_number(debug_status):
        print(message)


def string_time(time_spent, digits):
    seconds = round(time_spent % 60, digits)
    minutes = round(time_spent // 60) % 60
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
    TIME_START = time.time()
    time.sleep(1)
    time_end = print_time("Time spent :", DEBUG_LEVEL, DEBUG_STATUS,
                          TIME_START, 6)
