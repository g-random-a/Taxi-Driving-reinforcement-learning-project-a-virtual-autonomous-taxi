import csv
import codecs

# Open the CSV file with utf-16 encoding
with codecs.open("model_output.csv", "r", "utf-16") as f:
    csvread = csv.reader(f, delimiter='\n')
    next(csvread)  # Skip the header row if there is one

    lines_list = []
    passes = 0
    fails = 0
    rewards = []
    errors = []
    total_rewards = 0

    # Read lines from the CSV file into lines_list
    for line in csvread:
        lines_list.append(line[0])

    lines_iter = iter(lines_list)

    # Process each line in the list
    for line in lines_iter:
        if line == 'Environment.act(): Primary agent has reached destination!':
            passes += 1
        elif line == 'Environment.step(): Primary agent ran out of time! Trial aborted.':
            fails += 1
        elif line == 'Reward is':
            reward_line = next(lines_iter, None)
            if reward_line:
                rewards.append(float(reward_line))

    # Calculate total rewards and errors
    total_rewards = sum(rewards)

    for reward in rewards:
        if reward < 0:
            errors.append(reward)

    total_errors = sum(errors)

    # Print the results
    print("Your cab made {} successful trips, and {} late.".format(passes, fails))
    print("It also had a total rewards of {} and a total error amount of {}".format(total_rewards, total_errors))
