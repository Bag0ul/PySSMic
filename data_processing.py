import pandas as pd
import numpy as np
import pickle


def open_file(file_name):
    path = os.path.join(ROOT_DIR, "results", file_name)
    print("opening " + path)
    with open(path, 'rb') as f:
        res = pickle.load(f)
        contracts = res[0]
        profiles = res[1]
    return contracts, profiles


def rename_columns(contract):
    old_keys = ["id", "time", "time_of_agreement", "load_profile", "job_id", "producer_id"]
    new_keys = ['Contract ID', 'Contract start time', 'Contract agreement time', 'Energy used (Wh)', 'Consumer ID',
                'Producer ID']
    for i in range(0, len(old_keys)):
        contract[new_keys[i]] = contract.pop(old_keys[i])
    return contract

####### IDENTIFY HOUSEHOLD ########
def neigbourhood_to_household(contracts_input, profiles_input,grid_id='grid'):
    household_dict_list = []
    for contracts in contracts_input:
        df = pd.DataFrame(contracts)
        producers = list(df.producer_id.unique())
        if grid_id in producers:
            producers.remove(grid_id)
        consumers = list(df.job_id.unique())
        all_devices = producers + consumers

        household_dict = dict()
        for device in all_devices:
            before = device.index('[') + 1
            after = device.index(']')
            household_id = device[before:after]
            if household_id in household_dict:
                household_dict[household_id].append(device)
            else:
                household_dict[household_id] = [device]

        household_dict_list.append(household_dict)

    return household_dict_list
###################################

######### profiles to list ########
def rewrite_profiles(contracts,profiles):
    profiles_list = []
    for i in range(len(contracts)):
        profiles_run = []
        df = pd.DataFrame(contracts[i])
        producers = df['producer_id'].tolist()
        for producer in producers:
            if producer != 'grid':
                profiles_run.append(profiles[i][producer])
        profiles_list.append(profiles_run)
    return contracts, profiles_list
###################################


######## HANDLING AVERAGES ########
# Average values in a list with lists
def average_list_with_lists(list_of_listwithlists):
    list_of_lists = []
    transposed_list_of_listwithlists = map(list,zip(*list_of_listwithlists))
    print(transposed_list_of_listwithlists)
    for tl in transposed_list_of_listwithlists:
        transposed_tl = map(list,zip(*tl))
        temp_list = []
        for tll in transposed_tl:
            temp_list.append(np.mean(tll))
        list_of_lists.append(temp_list)
    return list_of_lists

# Average value for the values in a dict
def average_from_dict(list_of_dicts):
    new_dict = dict()
    for d in list_of_dicts:
        for key, value in d.items():
            if key in new_dict:
                new_dict[key].append(value)
            else:
                new_dict[key] = [value]

    final_dict = dict()
    for key in new_dict.keys():
        value_list = new_dict[key]
        value_list_transposed = map(list,zip(*value_list))
        new_value_list = []
        for vlt in value_list_transposed:
            new_value_list.append(np.mean(vlt))
        final_dict[key] = new_value_list

    return final_dict

###################################

###### PREPROCESS INPUT DATA ######
# Divides time index by 60 and gives opportunity to accumlate values
def change_index_time(df_energy_list, start_times, cumulative=True):
    i = 0
    energy_dict = dict()
    for df_energy in df_energy_list:
        index_list = df_energy.index.tolist()
        step_size = int((index_list[1] - index_list[0]) / 60)

        if step_size == 0:
            new_indeces = index_list
        elif step_size == 1:
            new_indeces = list(range(start_times[i], len(df_energy.index) + start_times[i]))
        else:
            new_indeces = list(range(start_times[i], (len(index_list) * step_size) + start_times[i], step_size))

        i += 1
        df_energy.index = new_indeces

        # Remove cumulation per load
        load_profile_list = df_energy.tolist()
        load_profile_flat = []
        j = 0
        for lp in load_profile_list:
            if j == 0:
                load_profile_flat.append(lp)
            else:
                load_profile_flat.append(lp - load_profile_list[j - 1])
            j += 1

        # Make dictionary of indices that actually exist and their value
        existing_indices_dict = dict()
        for k in range(len(new_indeces)):
            existing_indices_dict[new_indeces[k]] = load_profile_flat[k]

        # Make an index list with all minutes in there
        all_indeces = list(range(0, new_indeces[-1] + 1))
        load_profile_per_minute = []
        for index in all_indeces:
            if index in existing_indices_dict:
                load_profile_per_minute.append(existing_indices_dict[index])
            else:
                load_profile_per_minute.append(np.nan)

        # Make zero from all nan values before load profile began
        count = 0
        for lp in load_profile_per_minute:
            if np.isnan(lp) == False:
                break
            else:
                load_profile_per_minute[count] = 0.0
                count += 1

        # Write the output to a dataframe
        df_energy = pd.DataFrame(data=load_profile_per_minute, index=all_indeces)
        # Interpolate the missing values (linear)
        df_energy = df_energy.interpolate()

        # the new index is the time the simulator is running
        # use this index/time to save the energy consumed in that minute
        for index in all_indeces:
            if index in energy_dict:
                energy_dict[index] += df_energy.ix[index].values[0]
            else:
                energy_dict[index] = df_energy.ix[index].values[0]

    # Make an time and cumulative energy
    energy_tuples = list(energy_dict.items())
    energy_tuples = sorted(energy_tuples, key=lambda x: x[0])

    x_energy = []
    y_temp = []
    for x, y in energy_tuples:
        x_energy.append(x)
        y_temp.append(y)

    if cumulative == True:
        # Add cumulation for all loads bundled
        accum = 0
        y_energy = []
        for y in y_temp:
            accum += y
            y_energy.append(accum)

        return [x_energy, y_energy]
    else:
        return [x_energy, y_temp]

#Pie chart: remote versus local
def remote_versus_local(contracts, grid_id='grid'):
    # Load the contracts into a pandas dataframe
    df = pd.DataFrame.from_dict(contracts)

    #Add a column 'energy_used' for the total energy exchanged during
    # the contract (last value in load_profile)
    load_profiles = df['load_profile'].tolist()
    energy_used = []
    for load_profile in load_profiles:
        energy_used.append(load_profile.iloc[len(load_profile.index)-1])
    df['energy_used'] = energy_used

    # Get all unique consumer_ids
    consumer_ids = df['job_id'].tolist()
    unique_consumer_ids = list(set(consumer_ids))

    # Determine the local and remote energy used
    output_dict = dict()
    for consumer in unique_consumer_ids:
        # Find all contracts from a specific consumer
        df_consumer_contracts = df.loc[df['job_id'] == consumer]
        # Find all local contracts and sum over the energy
        df_consumer_local = df_consumer_contracts.loc[df_consumer_contracts['producer_id'] != grid_id]
        local_energy = df_consumer_local['energy_used'].sum()
        #Find all remote contracts and sum over the energy
        df_consumer_remote = df_consumer_contracts.loc[df_consumer_contracts['producer_id'] == grid_id]
        remote_energy = df_consumer_remote['energy_used'].sum()
        # Write the consumer_id to a dictionary with a list of his local and remote energy
        output_dict[consumer] = [local_energy, remote_energy]

    return output_dict

#Line chart: energy over time
def energy_over_time(contracts, producer_profiles):

    #CONSUMPTION
    # Load the contracts into a dataframe
    df = pd.DataFrame.from_dict(contracts)

    # Change the seconds into minutes (round downwards)
    agreement_times = df['time_of_agreement'].tolist()
    agreement_minutes = []
    for at in agreement_times:
        agreement_minutes.append(int(at/60))
    df['time_of_agreement_minutes'] = agreement_minutes

    x_consumption, y_consumption = change_index_time(df['load_profile'].tolist(),agreement_minutes)

    #PRODUCTION
    x_production, y_production = change_index_time(producer_profiles, [0]*len(producer_profiles))

    if x_consumption == [] and y_consumption == []:
        x_consumption = [0]
        y_consumption = [0]
    if x_production == [] and y_production == []:
        x_production = [0]
        y_production = [0]

    return [x_consumption, y_consumption, x_production, y_production]

# Peak to average ratio
def peak_to_average_ratio(contracts, producer_profiles):
    #Load the contracts into a dataframe
    df = pd.DataFrame.from_dict(contracts)
    print(df)

    #Change the seconds into minutes (round downwards)
    agreement_times = df['time_of_agreement'].tolist()
    agreement_minutes = []
    for at in agreement_times:
        agreement_minutes.append(int(at/60))
    df['time_of_agreement_minutes'] = agreement_minutes

    # Get consumptions per minute
    x_consumption, y_consumption = change_index_time(df['load_profile'].tolist(), agreement_minutes, False)
    print(x_consumption)
    # Determine the consumption per quarter (by summing up all consumptions per minute)
    consumption_per_quarter = []
    quarter_consumption = 0.0
    for i in range(len(x_consumption)):
        quarter_consumption += y_consumption[i]
        if x_consumption[i] % 15 == 0:
            consumption_per_quarter.append(quarter_consumption)
            quarter_consumption = 0

    # Delete first 0.0 empty, so it does not influence the consumption per quarter
    #if consumption_per_quarter[0] == 0.0:
    #    consumption_per_quarter = consumption_per_quarter[1:]

    # Determine average and peak consumption
    average_consumption = np.mean(consumption_per_quarter)
    max_consumption = max(consumption_per_quarter)

    return max_consumption/average_consumption

###################################
########## NEIGHBOURHOOD ##########
###################################
########### EXECUTION 1 ###########
def neigbourhood_execution_remote_versus_local(contracts_input,profiles_input):
    contracts_input, profiles_input = rewrite_profiles(contracts_input,profiles_input)
    output = []
    for i in range(len(contracts_input)):
        output.append(remote_versus_local(contracts_input[i]))
    output_combined = average_from_dict(output)
    return [output, output_combined]
# output, output_combined = neigbourhood_execution_remote_versus_local(contracts,profiles)
###################################
########### EXECUTION 2 ###########
def neigbourhood_execution_energy_over_time(contracts_input,profiles_input):
    contracts_input, profiles_input = rewrite_profiles(contracts_input,profiles_input)
    output = []
    for i in range(len(contracts_input)):
        output.append(energy_over_time(contracts_input[i],profiles_input[i]))
    output_combined = average_list_with_lists(output)
    return [output, output_combined]
# output, output_combined = neigbourhood_execution_energy_over_time(contracts,profiles)
# print(output)
# attempt_print(output,output_combined)
###################################
########### EXECUTION 3 ###########
def neigbourhood_execution_peak_to_average(contracts_input,profiles_input):
    contracts_input, profiles_input = rewrite_profiles(contracts_input,profiles_input)
    output = []
    for i in range(len(contracts_input)):
        output.append(peak_to_average_ratio(contracts_input[i],profiles_input[i]))
    output_combined = np.mean(output)
    return [output, output_combined]
# output, output_combined = neigbourhood_execution_peak_to_average(contracts,profiles)
###################################

############ HOUSEHOLD ############
###################################
########### EXECUTION 1 ###########
def household_execution_local_versus_remote(contracts_input,households,profiles_input):
    output = []
    households_list = []
    for i in range(len(contracts_input)):
        household_ids = households[i].keys()
        households_list.append(list(household_ids))
        output_household = []
        for household in household_ids:
            household_devices = households[i][household]
            contracts_household = []
            for j in range(len(contracts_input[i])):
                if contracts_input[i][j]['job_id'] in household_devices:
                    contracts_household.append(contracts_input[i][j])
            output_household.append(remote_versus_local(contracts_household))
        output.append(output_household)
    return output, households_list
# output, households_list = household_execution_local_versus_remote(contracts,household_dictionary,profiles)
###################################
########### EXECUTION 2 ###########
def household_execution_energy_over_time(contracts_input,profiles_input,households):
    output = []
    households_list = []
    for i in range(len(contracts_input)):
        household_ids = households[i].keys()
        households_list.append(list(household_ids))
        output_household = []
        for household in household_ids:
            household_devices = households[i][household]
            contracts_household = []
            for j in range(len(contracts_input[i])):
                if contracts_input[i][j]['job_id'] in household_devices or contracts_input[i][j]['producer_id'] in household_devices:
                    contracts_household.append(contracts_input[i][j])
            profile_keys = profiles_input[i].keys()
            profiles_household = []
            for key in profile_keys:
                if key in household_devices:
                    profiles_household.append(profiles_input[i][key])
            output_household.append(energy_over_time(contracts_household,profiles_household))
        output.append(output_household)
    return output,households_list
# output, households_list = household_execution_energy_over_time(contracts,profiles,household_dictionary)
###################################
########### EXECUTION 3 ###########
def household_execution_peak_to_average_ratio(contracts_input,profiles_input,households):
    output = []
    households_list = []
    for i in range(len(contracts_input)):
        household_ids = households[i].keys()
        households_list.append(list(household_ids))
        output_household = []
        for household in household_ids:
            household_devices = households[i][household]
            contracts_household = []
            for j in range(len(contracts_input[i])):
                if contracts_input[i][j]['job_id'] in household_devices or contracts_input[i][j]['producer_id'] in household_devices:
                    contracts_household.append(contracts_input[i][j])
            profile_keys = profiles_input[i].keys()
            profiles_household = []
            for key in profile_keys:
                if key in household_devices:
                    profiles_household.append(profiles_input[i][key])
            output_household.append(peak_to_average_ratio(contracts_household,profiles_household))
        output.append(output_household)
    return output,households_list
# output, households_list = household_execution_peak_to_average_ratio(contracts,profiles,household_dictionary)
###################################

