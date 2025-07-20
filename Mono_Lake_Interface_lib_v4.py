# Make Plots, Get Stats from Runoff Aggregation

import xarray as xr
import pandas as pd
import numpy as np
import math
import os, sys
import itertools
import ipywidgets as widgets
import distinctipy
import seaborn as sns

# reduce warnings (less pink)
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['axes.facecolor'] = 'white'
rcParams['axes.edgecolor'] = 'black'
rcParams['axes.titlesize'] = 18
rcParams['legend.fontsize'] = 14.5
rcParams['axes.labelsize'] = 14.5
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15

import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 9

# EXTRA MATPLOTLIB LIBRARIES
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter
from cycler import cycler
from matplotlib import cm
from matplotlib import rcParams
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
    
### Max Number of Years for Policy Phases 
### by defining seperately, allows for policy to be set-up for max_num_years possible and then budget model simulation for different numbers of years
Max_Num_Years = (2100-1955)+1

dict_of_variable_to_unit = {'Water_Level':'ft','Storage':'ac-ft','Exports':'ac-ft','Transition_Time':'Years'}
### how the dry vs wet year_types will be split
dry_types = ['Dry', 'Dry-Normal', 'Normal']
wet_types = ['Wet-Normal', 'Wet', 'Extreme-Wet']

# Initial_Policy = {
#         1:
#         {6376: [0]*6,
#          6377: [4000,4000,4000,0,0,0],
#          6380: [6000,6000,6000,0,0,0],
#          6386: [10000,10000,10000,0,0,0],
#          6391: [12000,12000,12000,0,0,0],
#          6393: [16000,16000,16000,8000,8000,0],
#          6395: [np.inf,np.inf,np.inf,10000,10000,0]
#          }
#     }

# Function to format tick labels with commas
def with_commas(x, pos):
    return "{:,}".format(int(x))

# A simple function for widget interaction
def f(x):
    print(x)
    return x

# Function to get a list of unique colors
def get_unique_colors(n):
    return distinctipy.get_colors(n)

# Function to configure user defined policy
def config_user_policy(num_phase):
    # Check if input is valid
    if not isinstance(num_phase, int) or num_phase < 1:
        sys.exit('Number of phase should be an integer and not less than 1.')
    
    tab_contents = ['Dry', 'Dry-Normal', 'Normal', 'Wet-Normal', 'Wet', 'Extreme-Wet']
    
    # Create a function to generate independent accordion widgets
    def create_phase_accordion():
        w1 = widgets.Text(
            description = 'Water Level Thresholds:',
            value = '6375 6377 6379',
            style = {'description_width': 'initial'},
            disabled = False,
            continuous_update = True
        )
        
        w2 = [widgets.Text(
            description = f'Export Amounts for Year Type:',
            value = '0 4500 8000' ,
            style = {'description_width': 'initial'},
            disabled = False,
            continuous_update = True
        ) for name in tab_contents]        
        tab = widgets.Tab()
        tab.children = w2
        for i in range(len(w2)):
            tab.set_title(i, tab_contents[i])
        
        section1 = widgets.Accordion(children=[w1])
        section1.set_title(0, "Set Water Level threshold (ft)")
        section2 = widgets.Accordion(children=[tab])
        section2.set_title(0, "Set Export Amount (ac-ft)")
        accordion = widgets.VBox([section1, section2])
      
        return accordion
    
    # Main tab for nesting
    tab_nest = widgets.Tab()
    
    if num_phase == 1:
        accordion = create_phase_accordion()
        tab_nest.children = [accordion]
        tab_nest.set_title(0, 'Phase 1')
    else:
        # Create tabs for multiple phases
        phase_tabs = [create_phase_accordion() for _ in range(num_phase)]
        for i, accordion in enumerate(phase_tabs):
            tab_nest.children = phase_tabs
            tab_nest.set_title(i, f'Phase {i + 1}')
        
        # # Add additional tabs for dynamic type and phase thresholds
        # w3 = widgets.Text(
        #     description='Type of dynamic:',
        #     value='year or level',
        #     style={'description_width': 'initial'},
        #     disabled=False,
        #     continuous_update=True
        # )
        
        # Replace the Text widget with a Dropdown widget
        w3 = widgets.Dropdown(
            options=['year', 'level'],  # Dropdown options
            value='year',  # Default value
            description='Type of dynamic:',
            style={'description_width': 'initial'},
            disabled=False
        )
        
        w4 = widgets.Accordion(children=[
            widgets.FloatText(
                description=f'Threshold value:',
                value=0,
                style={'description_width': 'initial'},
                disabled=False,
                continuous_update=True
            ) for _ in range(1, num_phase)
        ])
        
        for i in range(num_phase - 1):
            w4.set_title(i, f'Start threshold of Phase {i + 2}:')
        
        tab_nest.children = phase_tabs + [w3, w4]
        tab_nest.set_title(num_phase, 'Dyn_Type')
        tab_nest.set_title(num_phase + 1, 'Phase_Thresh')
    
    return tab_nest




# Define the linear extrapolation function
def polyfit_func(x, min1, max1, min2, max2, slope_s, slope_e, thresh, polyfit):
    
    '''
    If value smaller or larger than observed range, then use linear slope to extrapolate rather than 
    polynomial which can extrapolate in undesirable manner
    '''
    
    vals = np.where(x < min1,
                           min2 + slope_s * (x - min1),
                           np.where(x > max1,
                                    max2 + slope_e * (x - max1),
                                    polyfit(x)))
    
    vals = np.where(vals < thresh, thresh, vals)
    
    return vals


def create_dataarray(phases_data, Dyn_Type=None, Phase_Thresh=None):
    """
    Create an xarray DataArray with the specified dimensions.
    
    Parameters:
    phases_data (dict): A dictionary where keys are phase numbers and values are dictionaries
                        with levels as keys and lists representing the RYT values for 'level' and 'year_type'.
    Dyn_Type (str, optional): A string indicating the type of dynamic ('year', 'level', 'year_level').
                              Only needed if more than 1 phase is defined.
    Phase_Thresh (list, optional): A list of thresholds for the start of each phase.
                                   Only needed if more than 1 phase is defined.
    
    Returns:
    xr.DataArray: The created DataArray with dimensions ('phase', 'level', 'year_type') and attached attributes.
    """
    # Define year_type labels
    year_types = ['Dry', 'Dry-Normal', 'Normal', 'Wet-Normal', 'Wet', 'Extreme-Wet']
    
    # Determine the global minimum and maximum levels across all phases
    global_min_level = min(min(levels.keys()) for levels in phases_data.values())
    global_max_level = max(max(levels.keys()) for levels in phases_data.values()) + 1
    
    # Create a list to store each phase's data as a DataArray
    phase_dataarrays = []
    phase_numbers = sorted(phases_data.keys())
    
    for phase in phase_numbers:
        phase_info = phases_data[phase]
        sorted_levels = sorted(phase_info.keys())
        first_defined_level = sorted_levels[0]
        
        # Initialize data array (rows based on levels, columns based on year_types) with np.nan
        data = np.full((global_max_level - global_min_level, len(year_types)), np.nan)
        
        # Set all levels below the first defined level to user-specified values or leave as np.nan
        if global_min_level < first_defined_level:
            # If the user specified a level below the first threshold, use it
            if global_min_level in phase_info:
                data[:first_defined_level - global_min_level] = phase_info[global_min_level]
            else:
                # Otherwise, default to np.nan or explicitly zero (based on user logic)
                data[:first_defined_level - global_min_level] = [0] * len(year_types)
        
        # Apply the values to the levels
        for i, level in enumerate(sorted_levels):
            values = phase_info[level]
            next_level = sorted_levels[i + 1] if i + 1 < len(sorted_levels) else global_max_level
            # Apply values across range of current and next level
            data[level - global_min_level:next_level - global_min_level] = values
        
        levels = np.arange(global_min_level, global_max_level)
        
        # Create a DataArray for this phase
        da = xr.DataArray(data, coords=[levels, year_types], dims=['level', 'year_type'])
        phase_dataarrays.append(da)
    
    # Concatenate all phase DataArrays along a new 'phase' dimension
    data_array = xr.concat(phase_dataarrays, dim='phase')
    data_array['phase'] = phase_numbers
    
    # Determine attributes based on the number of phases
    if len(phase_numbers) == 1:
        attributes = {
            'Dynamic': 0,
            'Dyn_Type': np.nan,
            'Phase_Num': np.nan,
            'Phase_Thresh': np.nan
        }
    else:
        attributes = {
            'Dynamic': 1,
            'Dyn_Type': Dyn_Type,
            'Phase_Num': len(phase_numbers),
            'Phase_Thresh': Phase_Thresh
        }
    
    # Attach attributes to the DataArray
    data_array.attrs = attributes
    
    return data_array

# Define various DWP policies
def create_dwp_policy():
    ### Create and Put DWP Policies into Dictionary
    ### Note, here DWP policies created so highest water level exports continue for 6391 and above
    ### However, when running WBM can define if Post-Transition Policy or Another User-Defined Policy is Used
    ### (e.g. so can either have: i) continuation of policy, ii) post-transition policy or iii) other user-defined policy

    Policy_Data = {}

    ### Existing
    A1 = {
        1: 
        {6376: [0]*6,
         6377: [4500]*6,
         6380: [16000]*6,
         6391: [16000]*6
         }
    }

    data_array = create_dataarray(A1)
    Policy_Data['A1'] = data_array

    ### No Exports
    A2 = {
        1:
        {6376: [0]*6,
         6391: [0]*6
         }
    }

    data_array = create_dataarray(A2)
    Policy_Data['A2'] = data_array

    ### No Exports in Wet & Extreme Wet Years
    A3 = {
        1:
        {6376: [0]*6,
         6377: [4500,4500,4500,4500,0,0],
         6380: [16000,16000,16000,16000,0,0],
         6391: [16000,16000,16000,16000,0,0]
         }
    }

    data_array = create_dataarray(A3)
    Policy_Data['A3'] = data_array

    ### No Exports in Wet-Normal, Wet, Extreme Wet & Higher Mono Lake Threshold Levels for Exports
    ### No Exports in Wet & Extreme Wet Years
    A4 = {
        1:
        {6376: [0]*6,
         6381: [4500,4500,4500,0,0,0],
         6383: [16000,16000,16000,0,0,0],
         6391: [16000,16000,16000,0,0,0]
         }
    }

    data_array = create_dataarray(A4)
    Policy_Data['A4'] = data_array

    ### Phase 1: Existing Policy with Higher Mono Lake Threshold Levels for Exports
    ### Phase 2 (occurs at 6386 ft): Same with Higher Mono Lake Threshold Levels for Exports
    A5 = {
        1: 
        {6376: [0]*6,
         6381: [4500]*6,
         6383: [16000]*6,
         6391: [16000]*6
         },
        2:
        {6376: [0]*6,
         6384: [4500]*6,
         6386: [16000]*6,
         6391: [16000]*6
         }
    }

    Dyn_Type = 'level'  # 'year', 'level', 'year_level'
    Phase_Thresh = [6386]  # list of thresholds for start of each phase (years passed, level at which to enact change)        

    data_array = create_dataarray(A5, Dyn_Type, Phase_Thresh)
    Policy_Data['A5'] = data_array

    ### No Exports in Wet-Normal, Wet, Extreme Wet & lower exports during dry years
    A6 = {
        1:
        {6376: [0]*6,
         6377: [4500,4500,4500,0,0,0],
         6391: [4500,4500,4500,0,0,0]
         }
    }

    data_array = create_dataarray(A6)
    Policy_Data['A6'] = data_array

    ### No Exports in Wet-Normal, Wet, Extreme Wet & Higher Mono Lake Threshold Levels for Exports
    ### Phase 2: Same with Higher Mono Lake Threshold Levels for Exports
    A7 = {
        1: 
        {6379: [0]*6,
         6380: [4500,4500,4500,0,0,0],
         6384: [8000,8000,8000,0,0,0],
         6391: [8000,8000,8000,0,0,0]
         },
        2:
        {6383: [0]*6,
         6384: [4500,4500,4500,0,0,0],
         6386: [8000,8000,8000,0,0,0],
         6391: [8000,8000,8000,0,0,0]
         }
    }

    Dyn_Type = 'level'  # 'year', 'level', 'year_level'
    Phase_Thresh = [6386]  # list of thresholds for start of each phase (years passed, level at which to enact change)

    data_array = create_dataarray(A7, Dyn_Type, Phase_Thresh)
    Policy_Data['A7'] = data_array

    ### Post-Transition
    A8 = {
        1:
        {6387: [0]*6,
         6388: [10000]*6,
         6391: [np.inf]*6,
         }
    }

    data_array = create_dataarray(A8)
    Policy_Data['A8'] = data_array

    ### No Exports in Normal, Wet-Normal, Wet, Extreme Wet & Higher Mono Lake Threshold Levels for Exports
    ### Phase 2: occurs after 10 years
    ### Phase 3: occurs after another 5 years (15 years from start)
    ### as coded in estreams, idea here was to apply phase 2 for 10 years, and then phase 3 for 5 years...
    ### here we apply phase 1 after 10 years and then apply phase 3 after another 5 years

    A9 = {
        1: 
        {6382: [0]*6,
         6383: [1500,1500,0,0,0,0],
         6391: [1500,1500,0,0,0,0]
         },
        2:
        {6382: [0]*6,
         6383: [1500,1500,0,0,0,0],
         6385: [3500,3500,0,0,0,0],
         6391: [3500,3500,0,0,0,0]
         },
        3:
        {6382: [0]*6,
         6383: [1500,1500,0,0,0,0],
         6388: [3500,3500,0,0,0,0],
         6391: [3500,3500,0,0,0,0]
         }
    }

    Dyn_Type = 'year'  # 'year', 'level', 'year_level'
    Phase_Thresh = [10,15]  # list of thresholds for start of each phase (years passed, level at which to enact change)

    data_array = create_dataarray(A9, Dyn_Type, Phase_Thresh)
    Policy_Data['A9'] = data_array

    #### Existing Policy with dynamic shifts at 10 and 20 years 
    A10 = {
        1: 
        {6376: [0]*6,
         6377: [4500]*6,
         6391: [4500]*6
         },
        2:
        {6376: [0]*6,
         6377: [4500]*6,
         6380: [16000]*6,
         6391: [16000]*6
         },
        3:
        {6376: [0]*6,
         6377: [4500]*6,
         6380: [8000]*6,
         6391: [8000]*6
         }
    }

    Dyn_Type = 'year'  # 'year', 'level', 'year_level'
    Phase_Thresh = [10,20]  # list of thresholds for start of each phase (years passed, level at which to enact change)

    data_array = create_dataarray(A10, Dyn_Type, Phase_Thresh)
    Policy_Data['A10'] = data_array
    
    return Policy_Data


# DISPLAY EXPORTS ALLOWED FOR EACH WATER LEVEL
def df_of_policy(policy):

    # Convert the DataArray to a DataFrame
    df = policy.to_dataframe(name='Value')

    # Reset the index to make the data more user-friendly
    df.reset_index(inplace=True)

    df = df.rename(columns={'level':'Water Level','year_type':'RYT'})

    # Display the DataFrame in a format that's easy to read
    df_pivot = df.pivot(index='Water Level', columns='RYT', values='Value')
    
    # Preserve the order of the original 'year_type' from the DataArray
    original_order = policy.coords['year_type'].values  # Get the original order from the DataArray
    df_pivot = df_pivot.reindex(columns=original_order)  # Reorder columns in the pivoted DataFrame

    return df_pivot

# Function to aggregate consecutive rows with identical values across all columns
def aggregate_consecutive_rows(df):
    
    # Create an empty DataFrame to store the results
    result = pd.DataFrame(columns=df.columns)
    # Initialize tracking variables
    start = df.index[0]
    prev_row = df.iloc[0]
    
    for i in range(1, len(df)):
        current_row = df.iloc[i]
        # If the current row is different from the previous row, record the range
        if not current_row.equals(prev_row):
            if start == df.index[i-1]:
                index_label = f"{start}"
            else:
                index_label = f"{start}–{df.index[i-1]}"
            result.loc[index_label] = prev_row
            start = df.index[i]
        prev_row = current_row
    
    # Add the last group
    if start == df.index[-1]:
        index_label = f"{start}"
    else:
        index_label = f"{start}–{df.index[-1]}"
    result.loc[index_label] = prev_row

    return result

def return_for_multiple_phases(policy):

    aggregated_phase_dfs = {}

    for phase in policy.phase:

        df = policy.sel(phase=phase)

        # get df of policy
        df = df_of_policy(df)

        # Aggregate rows
        aggregated_df = aggregate_consecutive_rows(df)

        # Dynamically identify the minimum and maximum water levels
        min_water_level = df.index.min()
        max_water_level = df.index.max()

        # Update the first and last index with <=Min and >=Max
        aggregated_df.index = aggregated_df.index.str.replace(f'{min_water_level}', f'≤{min_water_level}')
        #aggregated_df.index = aggregated_df.index.str.replace(f'{max_water_level}–{max_water_level}', f'≥{max_water_level}')

        # update name for rows/col
        aggregated_df.columns.name = 'Water Level / RYT'

        int_phase = int(phase.values)
        
        # Store the aggregated DataFrame for this phase
        aggregated_phase_dfs[int_phase] = aggregated_df
        
    ### if single phase then just return df
    if int_phase == 1:
        aggregated_phase_dfs = aggregated_phase_dfs[1]

    return aggregated_phase_dfs

def dynamic_shift_dataarray(phases_data, years_between_phases, shift_levels, stop_level, max_num_years, Dyn_Type=None, reduction_percent=0):
    """
    Create an xarray DataArray policy
    
    Can include following dynamic conditions (individually or as a combination):
    
    i) percent reduction in values for each subsequent phase (e.g. 10% reduction every 5 years)
    ii) shift in water level threshold (e.g. shift water level threshold by 2 ft. every 20 years)
    
    Parameters:
    phases_data (dict): A dictionary where keys are phase numbers and values are dictionaries
                        with levels as keys and lists representing the RYT values for 'level' and 'year_type'.
    Dyn_Type (str, optional): A string indicating the type of dynamic ('year', 'level', 'year_level').
                              Only needed if more than 1 phase is defined.
    reduction_percent (float): The percentage by which to reduce the values in each subsequent phase.
    years_between_phases (int): The number of years between each phase.
    shift_levels (int): The number of levels to shift the values upward every `years_between_phases`.
    
    Returns:
    xr.DataArray: The created DataArray with dimensions ('phase', 'level', 'year_type') and attached attributes.
    """
    # Number of phases and when they occur
    phase_thresh = list(range(0, max_num_years, years_between_phases))
    Num_Phases = len(phase_thresh)
    
    # Define year_type labels
    year_types = ['Dry', 'Dry-Normal', 'Normal', 'Wet-Normal', 'Wet', 'Extreme-Wet']
    
    # Determine the global minimum and maximum levels across all phases
    global_min_level = min(min(levels.keys()) for levels in phases_data.values())
    global_max_level = max(max(levels.keys()) for levels in phases_data.values()) + 1
    
    # Adjust the global maximum level to accommodate the upward shifts
    global_max_level += Num_Phases * shift_levels
    
    # Create a list to store each phase's data as a DataArray
    phase_dataarrays = []
    initial_phase_data = phases_data[1]
    
    for phase in range(Num_Phases):
        sorted_levels = sorted(initial_phase_data.keys())
        first_defined_level = sorted_levels[0]
        
        # Initialize data array with np.nan
        data = np.full((global_max_level - global_min_level, len(year_types)), np.nan)
        
        # Set all levels below the first defined level to zero
        zero_values = [0] * len(year_types)
        if first_defined_level > global_min_level:
            data[:first_defined_level - global_min_level] = zero_values
        
        for i, level in enumerate(sorted_levels):
            # For levels >= stop_level (6392 ft), retain the original values from the initial policy
            if level >= stop_level:
                values = initial_phase_data[level]
            else:
                # Apply dynamic changes (shift, reduction) only for levels < stop_level
                values = [v * (1 - reduction_percent / 100) ** phase for v in initial_phase_data[level]]

            next_level = sorted_levels[i + 1] if i + 1 < len(sorted_levels) else global_max_level
            data[level - global_min_level:next_level - global_min_level] = values
        
        # Apply the upward shift only for levels < stop_level
        data_shifted = np.full_like(data, np.nan)
        shift_index = phase * shift_levels
        
        if shift_index > 0:
            # For levels < stop_level, apply shifts
            for idx in range(len(data)):
                actual_level = global_min_level + idx
                if actual_level < stop_level:
                    data_shifted[shift_index + idx] = data[idx]
                else:
                    # Retain original values for levels >= stop_level
                    data_shifted[idx] = data[idx]
                    
            # Set values below the first defined level to zero (handle lowest water levels)
            data_shifted[:shift_index] = zero_values
        else:
            data_shifted = data
        
        levels = np.arange(global_min_level, global_max_level)
        
        # Create a DataArray for this phase
        da = xr.DataArray(data_shifted, coords=[levels, year_types], dims=['level', 'year_type'])
        phase_dataarrays.append(da)
    
    # Concatenate all phase DataArrays along a new 'phase' dimension
    data_array = xr.concat(phase_dataarrays, dim='phase')
    data_array['phase'] = np.arange(1, Num_Phases + 1)
    
    # Determine attributes based on the number of phases
    if Num_Phases == 1:
        attributes = {
            'Dynamic': 0,
            'Dyn_Type': np.nan,
            'Phase_Num': np.nan,
            'Phase_Thresh': np.nan
        }
    else:
        attributes = {
            'Dynamic': 1,
            'Dyn_Type': Dyn_Type,
            'Phase_Num': Num_Phases,
            'Phase_Thresh': phase_thresh,
            "Stop_Level": stop_level
        }
        
    # Attach attributes to the DataArray
    data_array.attrs = attributes
    
    return data_array

### Dynamic Policy with Exponential Change in Percent Reduction 
### (So Higher Percent Reduction Applied at Lower Water Levels Compared to Higher Water Levels)
def reduction_curve(level, first_export_level, stop_level, max_reduction, min_reduction):
    """
    Calculate the reduction percentage based on the water level using a non-linear curve,
    starting from the first water level that allows exports and stopping at the stop_level.
    
    Parameters:
    level (int): The current water level.
    first_export_level (int): The first level that allows exports.
    stop_level (int): The stop level for dynamic modifications.
    max_reduction (float): The maximum reduction percentage.
    min_reduction (float): The minimum reduction percentage.
    
    Returns:
    float: The reduction percentage for the given water level.
    """
    range_levels = stop_level - first_export_level
    if level < first_export_level or level > stop_level:
        return 0  # No reduction applied outside the defined range
    
    level_position = (level - first_export_level) / range_levels
    reduction = max_reduction * np.exp(-3 * level_position) + min_reduction * (1 - np.exp(-3 * level_position))
    return reduction

def dynamic_dataarray(phases_data, years_between_phases, stop_level, max_num_years, Dyn_Type=None, reduction_percent=0, use_non_linear=False, min_reduction_percent=0):
    """
    Create an xarray DataArray with dynamic conditions.
    
    Can include the following dynamic conditions (individually or as a combination):
    
    i) percent reduction in values for each subsequent phase (e.g. 10% reduction every 5 years)
    
    Parameters:
    phases_data (dict): A dictionary where keys are phase numbers and values are dictionaries
                        with levels as keys and lists representing the RYT values for 'level' and 'year_type'.
    Dyn_Type (str, optional): A string indicating the type of dynamic ('year', 'level', 'year_level').
                              Only needed if more than 1 phase is defined.
    reduction_percent (float): The percentage by which to reduce the values in each subsequent phase.
    years_between_phases (int): The number of years between each phase.
    
    Returns:
    xr.DataArray: The created DataArray with dimensions ('phase', 'level', 'year_type') and attached attributes.
    """
    
    # Number of phases and when they occur
    phase_thresh = list(range(0, max_num_years, years_between_phases))
    Num_Phases = len(phase_thresh)

    # Define year_type labels
    year_types = ['Dry', 'Dry-Normal', 'Normal', 'Wet-Normal', 'Wet', 'Extreme-Wet']

    # Determine the global minimum and maximum levels across all phases
    global_min_level = min(min(levels.keys()) for levels in phases_data.values())
    global_max_level = max(max(levels.keys()) for levels in phases_data.values()) + 1

    # Find the first level that allows exports (has non-zero values)
    first_export_level = next(level for level in sorted(phases_data[1].keys()) if any(phases_data[1][level]))

    # Create a list to store each phase's data as a DataArray
    phase_dataarrays = []
    initial_phase_data = phases_data[1].copy()

    for phase in range(Num_Phases):
        sorted_levels = sorted(initial_phase_data.keys())
        first_defined_level = sorted_levels[0]

        # Initialize data array with np.nan
        data = np.full((global_max_level - global_min_level, len(year_types)), np.nan)

        # Set all levels below the first defined level to zero
        zero_values = [0] * len(year_types)
        if first_defined_level > global_min_level:
            data[:first_defined_level - global_min_level] = zero_values

        # Apply the values to the levels
        for i, level in enumerate(sorted_levels):
            values = initial_phase_data[level]
            next_level = sorted_levels[i + 1] if i + 1 < len(sorted_levels) else global_max_level
            data[level - global_min_level:next_level - global_min_level] = values

    #     if phase == 2:
    #         sys.exit('test')

        # Apply reduction for this phase
        if phase > 0:
            for level in range(global_min_level, global_max_level):
                if level >= stop_level:
                    break
                if use_non_linear:
                    reduction = reduction_curve(level, first_export_level, stop_level, reduction_percent, min_reduction_percent)
    #                 if phase == 1:
    #                     print(f'{level}: {reduction}')
    #                     if level == 6388:
    #                         tmp = level - global_min_level
    #                         print(f'{tmp}')   
    #                         sys.exit('test')
                    data[level - global_min_level] *= (1 - reduction / 100)
                else:
                    data[level - global_min_level] *= (1 - reduction_percent / 100)

        levels = np.arange(global_min_level, global_max_level)


        # Create a DataArray for this phase
        da = xr.DataArray(data, coords=[levels, year_types], dims=['level', 'year_type'])
        phase_dataarrays.append(da)

        # Update initial_phase_data to be the data for the next phase calculation
        #initial_phase_data = {level: list(data[level - global_min_level]) for level in sorted_levels}
        initial_phase_data = {level: list(data[level-global_min_level]) for level in levels}


    # Concatenate all phase DataArrays along a new 'phase' dimension
    data_array = xr.concat(phase_dataarrays, dim='phase')
    data_array['phase'] = np.arange(1, Num_Phases + 1)

    # Determine attributes based on the number of phases
    if Num_Phases == 1:
        attributes = {
            'Dynamic': 0,
            'Dyn_Type': np.nan,
            'Phase_Num': np.nan,
            'Phase_Thresh': np.nan
        }
    else:
        attributes = {
            'Dynamic': 1,
            'Dyn_Type': Dyn_Type,
            'Phase_Num': Num_Phases,
            'Phase_Thresh': phase_thresh,
            "Stop_Level": stop_level
        }

    # Attach attributes to the DataArray
    data_array.attrs = attributes
    
    return data_array

# Define user policies
def create_user_policy(policy_data, max_num_years):
    
    # Example usage with reduction_percent, years_between_phases, and shift_levels
    reduction_percent = 10  # same reduction to all water levels
    years_between_phases = 5
    Dyn_Type = 'year_level'
    shift_levels = 0  # number of levels to shift upward
    stop_level = 6391  ### shifts in policy stop if this water level is met and dynamic shifts only applied up to this level

    ### Run Function to Create User Defined Policy
    Shift_Level_and_Exports = dynamic_shift_dataarray(Initial_Policy,  
                                         years_between_phases, shift_levels, stop_level, max_num_years,
                                         Dyn_Type, reduction_percent)

    ### Add User Defined Policy to Dictionary of Policies
    policy_data['U1'] = Shift_Level_and_Exports
    
    # Example usage with reduction_percent, years_between_phases, and shift_levels
    reduction_percent = 0  # same reduction to all water levels
    years_between_phases = 5
    Dyn_Type = 'year_level'
    shift_levels = 1  # number of levels to shift upward
    stop_level = 6391

    ### Run Function to Create User Defined Policy
    Shift_Level_and_Exports = dynamic_shift_dataarray(Initial_Policy,  
                                         years_between_phases, shift_levels, stop_level, max_num_years,
                                         Dyn_Type, reduction_percent)

    ### Add User Defined Policy to Dictionary of Policies
    policy_data['U2'] = Shift_Level_and_Exports

    # Example usage with reduction_percent, years_between_phases, and shift_levels
    reduction_percent = 10  # same reduction to all water levels
    years_between_phases = 10
    Dyn_Type = 'year_level'
    shift_levels = 1  # number of levels to shift upward
    stop_level = 6391

    ### Run Function to Create User Defined Policy
    Shift_Level_and_Exports = dynamic_shift_dataarray(Initial_Policy,  
                                         years_between_phases, shift_levels, stop_level, max_num_years,
                                         Dyn_Type, reduction_percent)

    ### Add User Defined Policy to Dictionary of Policies
    policy_data['U3'] = Shift_Level_and_Exports
    
    # Example usage with reduction_percent, years_between_phases, and shift_levels
    reduction_percent = 20  # max reduction percent
    years_between_phases = 5
    Dyn_Type = 'year_level'
    stop_level = 6391  ### also used to define at what water level dynamic modifications stop

    # Use non-linear reduction
    use_non_linear = True  # use non-linear reduction
    min_reduction_percent = 0  # min reduction percent for non-linear reduction

    ### Run Function to Create User Defined Policy
    Exponential_Perc_Reduction = dynamic_dataarray(Initial_Policy, 
                                                        years_between_phases, stop_level, max_num_years, 
                                                        Dyn_Type, reduction_percent, use_non_linear, min_reduction_percent)

    ### Add User Defined Policy to Dictionary of Policies
    policy_data['U4'] = Exponential_Perc_Reduction
    
    return policy_data

# Get a list of GCMs
def get_gcms(gcm_data_path):
    
    ### Load GCMs That Are Available
    GCMs = os.listdir(f'{gcm_data_path}/Dynamic_RYT_SEF')

    ### Filter By GCMs that Cover all 3 SSPs
    GCM_df = pd.DataFrame(GCMs,columns=['GCM_SSP'])
    GCM_df = GCM_df.sort_values(by='GCM_SSP')
    GCM_df = GCM_df.reset_index(drop=True)

    # Extract model names and SSPs
    GCM_df['Model'] = GCM_df['GCM_SSP'].str.extract(r'(^.*?)_r')
    GCM_df['SSP'] = GCM_df['GCM_SSP'].str.extract(r'(ssp\d+)')
    GCM_df['Member'] = GCM_df['GCM_SSP'].str.extract(r'^[^_]*_([^_]*)_')

    # Group by model and filter based on the count of unique SSPs
    Filtered_GCM_df = GCM_df.groupby('Model').filter(lambda x: x['SSP'].nunique() == 3)

    number_of_GCMs_with_all_SSPs = Filtered_GCM_df['Model'].nunique()

    #print(f'{number_of_GCMs_with_all_SSPs} GCMs Exist for All 3 SSPs')

    GCMs = Filtered_GCM_df['GCM_SSP'].values

    # Remove everything after "_ssp" and remove duplicates
    GCM_Member_List = list(np.unique(np.array([gcm.split('_ssp')[0] for gcm in GCMs])))

    ### Also Create List that Can Easily Reference with Simple GCM Names
    GCM_Names_Only = Filtered_GCM_df['Model'].unique()

    simple_GCM_Name_List = []

    for gcm in GCM_Names_Only:

        gcm_simple_name = gcm.split('_')[0]

        simple_GCM_Name_List.append(gcm_simple_name)
    
    return (simple_GCM_Name_List, GCM_Member_List)

### Function to Determine Export Limit for any Input Policy ###

## Outputs the export limit based on policy xarray and conditions fed in from the model for each year ##
## For Static Policy: Depends on Lake Level and RYT
## For Dynamic Policy: Additionally Depends on Years into Simulation and Lake Level Condition to Determine Phase

## Function Inputs ##

# Policy_Data : xarray for specified policy (either user defined policy or existing LADWP policies)
# policy_name : name of LADWP policy (A1-A10) or "NA" for user defined policy 
# level : input lake level from model 
# year : input current year from model
# year_type_check : input RYT from model
# level_hist : list of all previous lake levels from 2024 - current year 

## Function Outputs ##
# export threshold value

def find_policy_value_updated(Policy_Data,level, year, start_year, year_type_check, level_hist, policy_name=None):
    
    '''
    Policy_Data can be an xarray datarray or dataset
    
    If dataset, then need to define policy_name so can extract the policy (as a variable) from the dataset
    
    '''
    
    #selects policy 
    if policy_name == None:
        if isinstance(Policy_Data, xr.DataArray):
            policy = Policy_Data
        else:
            sys.exit('Need to Name Policy To Extract from Dataset')
    else:
        policy = Policy_Data[policy_name]#Policy_Data[policy_name]

    #looks up policy attributes to check if its a dynamic policy 
    policy_attr = policy.attrs
    check_dyn = policy_attr["Dynamic"]

    #Caps upper and lower bounds of lake level ranges
    #note, this is only relevant for determining exports
    
    ### use exports defined at lowest level if below that level (would be zero exports)
    if level < float(np.min(policy.level).values):
        level = float(np.min(policy.level).values)
    ### use exports defined at highest level if above that level
    if level > float(np.max(policy.level).values):
        level = float(np.max(policy.level).values)

    #Calculates years passed to use in dynamic policies based on time
    years_passed = year - start_year

    #If the policy is dynamic....
    if check_dyn == 1:

        #checks policy attributes 
        dyn_type = policy_attr['Dyn_Type'] #whether policy dynamic in lake level or time
        phase_num = policy_attr['Phase_Num'] #how many phases the policy has
        phase_thresh = policy_attr['Phase_Thresh'] # list of phase thresholds for time or lake level 

        ##Changes phase number based on threshold##
        phase_check = 1

        # for dynamic policies based on time (will not check if a water level has been met or not!)
        # A9 and A10
        if dyn_type == "year":
            for i in range(phase_num-1):
                if years_passed > phase_thresh[i]:
                    phase_check+=1

        # this is the main dynamic condition used right now
        # will check if a water level has been met or not!
        # if not, then will shift to next phase if user-defined years have passed
        # note, once stop_level (e.g. 6391) is reached will stop shifting phases
        elif dyn_type == "year_level":
            level_stop = policy_attr["Stop_Level"]
            dyn_level_check = max(level_hist)
            if dyn_level_check >= level_stop:
                ### check years passed since maximum water level above level_stop (e.g. 6391 ft.)
                ### this will then be used to select policy phase below
                ### removed +1 to line of code below
                years_passed = (level_hist.index(dyn_level_check))
            for i in range(phase_num-1):
                ### here loops from 0 to max number of phases
                ### issue appears that years_passed will be greater than first phase value of zero
                ### added i+1 to fix this issue and changed > to >=
                if years_passed >= phase_thresh[i+1]:
                    phase_check+=1

        #for dynamic policies based on level 
        #New phase triggered when lake level reached a certain level
        #check if level has ever been reached 
        # A7
        elif dyn_type == "level":
            dyn_level_check = max(level_hist)
            for i in range(phase_num-1):
                if dyn_level_check >= phase_thresh[i]:
                    phase_check+=1

        #rounds level down into integer for export selection 
        level_check = math.floor(level)

        #selects export value for particular phase, level and year type
        #note, first phase = 1 (why phase_check is set equal to 1)
        export = policy.sel(phase=phase_check).sel(level=level_check).sel(year_type=year_type_check)
    
    #If the policy is not dynamic...
    else:
        #rounds level down into integer for export selection 
        level_check = math.floor(level)
        #selects export value level and year type
        export = policy.sel(level=level_check).sel(year_type=year_type_check)
    
    #returns export limit 
    return export.values

### Function that Runs Model for GCM and Policy Inputs ###

# Left mostly the same with updated policy implementation using find_policy_value_updated()

def predict_Mono_Lake_Water_Level_Added_Policies(######### FILE LOCATIONS ##########
                                                 model_file,
                                                 lake_file,
                                                 ######### NEVER MODIFY ############
                                                 measured_Lee_Vining,
                                                 measured_Walker_Parker,
                                                 measured_inflow_not_managed_by_DWP,
                                                 measured_Grant_Lake_Spillway_Outflow,
                                                 Mono_Lake_P,
                                                 Mono_Lake_PET,
                                                 Ungauged_Runoff,
                                                 Year_Type,
                                                 Amended_License_Flow_Requirement,
                                                 ##### THESE ARE FLEXIBLE USER-DEFINED OPTIONS ###
                                                 Initial_Water_Level,
                                                 Start_Year, 
                                                 End_Year,
                                                 Export_Policy,
                                                 Post_Transition_Policies, ### UPDATED THIS SO LOOPS THROUGH POST-TRANSITION POLICIES
                                                 # Post_Transition_Policy,
                                                 # User_Post_Transition_Policy,
                                                 Policy_Name=None,
                                                 Post_Transition_Policy_to_Use=None):
    
    '''
    Year_Type = includes information regarding runoff year type for each year based on 4-creek runoff
    
    measured variables represent actual flow after regulation
    and can be based on observed data or FNF-->Predicted Regulated Flow
    
    Post_Transition_Policy_to_Use = Names of Post-Transition Policies (e.g. 'None','D_1631','User_Defined')
    
    Note, Export_Policy and Post_Transition_Policies are the dictionaries of the pre- and post-transition policies
    These go into the function "find_policy_value_updated" which also requires the pre- and post-transition
    policy name which is included in this function as: "Policy_Name" and "Post_Transtion_Policy_to_Use"
    
    '''
    
    ### First Get Information Needed for Mono Lake Level Model

    #details_for_model = pd.read_csv('/data/public/Mono_Lake/DATA/3_Model_Data/details_for_model.csv',index_col=0)
    details_for_model = pd.read_csv(model_file,index_col=0)

    #### this is accounted for in Lake Level Model Notebook for both GCMs and ERA5 flows into mono lake
    #### optimized bias correction term
    Sim_Modification_to_Ungauged_Runoff = details_for_model[details_for_model.index=='Sim_Modification_to_Ungauged_Runoff'].values[0][0]

    #print('If update water level model, make sure to update error in storage AND BC to Ungauged for ERA5')
    ### Error Term for Observed Data
    intercept_error_using_observed_inflow = details_for_model[details_for_model.index=='intercept_error_using_observed_inflow'].values[0][0]
    slope_error_using_observed_inflow =  details_for_model[details_for_model.index=='slope_error_using_observed_inflow'].values[0][0]
    
    
    ### Additional Details for Model

    GW_Export = 5500

    min_salinity_level = 6360
    min_salinity_value = 0.93
    max_salinity_level = 6405
    max_salinity_value = 0.97

    ### Data needed if want to have variable ungauged surface area
    Ungauged_Assumed_Area = 1299.0291932258256 * 247.105 ### Acres
    Mono_SHP_Area = 39841.920314190225 ### Acres ### SHP Area for Mono Lake (corresonds to 6374 ft)
    
    ### assuming updates to ungauged surface area based on mono lake surface area are included
    #Include_SA_Updates_To_Ungauged = True  ### this is actually decided in 5_Mono_Lake_Level_Model_vX
    
    
    #####################################################################

    ### Modification to Mono Lake Evaporation based on salinity at different lake elevations

    ### Assume a salinity-elevation relationship
    m_salinity = (max_salinity_value-min_salinity_value) / (max_salinity_level-min_salinity_level)  ### corresponding salinity / corresponding elevations
    b_salinity = min_salinity_value - m_salinity * min_salinity_level

    ### load observed data range of elev-vol-sa
    #mono_lake_storage_elev = pd.read_csv('/data/public/Mono_Lake/DATA/Lake_Volume_Elevation/Mono_Lake/Mono_Lake_Area_Storage_Elev.txt',delimiter='\t')
    mono_lake_storage_elev = pd.read_csv(lake_file,delimiter='\t')

    ### Get fit between elevation and storage/surface area
    df = mono_lake_storage_elev.copy()
    df = df.sort_values('elev_ft')  ### also sorts volume

    # Fit a quadratic polynomial to the data
    obs_polyfit_elev_storage = np.polyfit(df['elev_ft'], df['volume_ac_ft'], 2)
    # get coefficients so can apply model
    obs_polyfit_elev_storage = np.poly1d(obs_polyfit_elev_storage)

    # Calculate the slopes based on the first two and last two values of the observed data
    obs_elev_storage_slope_start = (df['volume_ac_ft'].iloc[1] - df['volume_ac_ft'].iloc[0]) / (df['elev_ft'].iloc[1] - df['elev_ft'].iloc[0])
    obs_elev_storage_slope_end = (df['volume_ac_ft'].iloc[-1] - df['volume_ac_ft'].iloc[-2]) / (df['elev_ft'].iloc[-1] - df['elev_ft'].iloc[-2])

    # use intercept at y_min, y_max
    storage_obs_min = np.min(df['volume_ac_ft'])
    storage_obs_max = np.max(df['volume_ac_ft'])

    elev_obs_min = np.min(df['elev_ft'])
    elev_obs_max = np.max(df['elev_ft'])

    ### from above slope extended from observed data, determine at what elevation of mono lake the volume is zero
    min_mono_lake_elevation = elev_obs_min - (storage_obs_min / obs_elev_storage_slope_start)

    # Fit a quadratic polynomial to the data
    obs_polyfit_storage_elev = np.polyfit(df['volume_ac_ft'], df['elev_ft'], 3)
    # get coefficients so can apply model
    obs_polyfit_storage_elev = np.poly1d(obs_polyfit_storage_elev)

    # Calculate the slopes based on the first two and last two values of the observed data
    obs_storage_elev_slope_start = (df['elev_ft'].iloc[1] - df['elev_ft'].iloc[0]) / (df['volume_ac_ft'].iloc[1] - df['volume_ac_ft'].iloc[0])
    obs_storage_elev_slope_end = (df['elev_ft'].iloc[-1] - df['elev_ft'].iloc[-2]) / (df['volume_ac_ft'].iloc[-1] - df['volume_ac_ft'].iloc[-2])

    ############ Fit Function Using Volume to Surface Area of Mono Lake

    ### from above slope extended from observed data, determine at what elevation of mono lake the volume is zero
    min_mono_lake_elevation = elev_obs_min - (storage_obs_min / obs_elev_storage_slope_start)

    # Fit a quadratic polynomial to the data
    obs_polyfit_storage_sa = np.polyfit(df['volume_ac_ft'], df['surface_area_acres'], 8)
    # get coefficients so can apply model
    obs_polyfit_storage_sa = np.poly1d(obs_polyfit_storage_sa)

    # Calculate the slopes based on the first two and last two values of the observed data
    obs_storage_sa_slope_start = (df['surface_area_acres'].iloc[1] - df['surface_area_acres'].iloc[0]) / (df['volume_ac_ft'].iloc[1] - df['volume_ac_ft'].iloc[0])
    obs_storage_sa_slope_end = (df['surface_area_acres'].iloc[-1] - df['surface_area_acres'].iloc[-2]) / (df['volume_ac_ft'].iloc[-1] - df['volume_ac_ft'].iloc[-2])

    ### get min/max of surface area in observed data
    sa_obs_min = np.min(df['surface_area_acres'])
    sa_obs_max = np.max(df['surface_area_acres'])
    
    ###################################################################################################################################################
    
    Reached_6391 = False

    dict_of_wsel_results = {}

    ### User-Defined, but Default = 2024 water level
    Current_Water_Level = Initial_Water_Level

    ### Get initial storage: Convert Water Level to Storage
    Current_Storage = polyfit_func(Current_Water_Level, elev_obs_min, elev_obs_max, storage_obs_min, storage_obs_max, obs_elev_storage_slope_start, obs_elev_storage_slope_end, 0, obs_polyfit_elev_storage)
    
    ### Need to multiply P, PET (inches) by Surface Area of Mono Lake to get Total ac-ft
    Current_Surface_Area = polyfit_func(Current_Storage, storage_obs_min, storage_obs_max, sa_obs_min, sa_obs_max, obs_storage_sa_slope_start, obs_storage_sa_slope_end, 0, obs_polyfit_storage_sa)
    
    ## ungauged area should be multiplied by this for first time-stamp
    Current_Ungauged_SA = Ungauged_Assumed_Area + (Mono_SHP_Area - Current_Surface_Area)

    ### For validating result
    Predicted_Elevation = []
    Predicted_Elevation.append(Current_Water_Level)

    ### for keeping track of storage
    Predicted_Storage = []
    Predicted_Storage.append(float(Current_Storage))
    
    ### these will be determined each year based on water levels and policies defined
    Exports_list = []
    Rush_into_Mono_list = []
    
    ### Now get ∆ Storage for each runoff year
    ### first prediction is for April 1st of following year after "Initial_Year"
    for year in range(Start_Year,End_Year+1):
        
        ### get current years regulated inflow
        measured_inflow_for_year = measured_inflow_not_managed_by_DWP[year] + measured_Lee_Vining[year] + measured_Walker_Parker[year]
        Grant_Spill_and_Outflow_for_year = measured_Grant_Lake_Spillway_Outflow[year]
        
        ############################################## Apply POLICY ##########################################
        
#         if (Reached_6391 and Post_Transition_Policy_to_Use in ['D_1631','User_Defined']):
            
            
#             if Post_Transition_Policy_to_Use == 'D_1631':
#                 export_allowed_for_year = find_policy_value_updated(Post_Transition_Policy, Current_Water_Level, 
#                                                                     year, Start_Year, Year_Type[year], Predicted_Elevation)
#             elif Post_Transition_Policy_to_Use == 'User_Defined':
#                 export_allowed_for_year = find_policy_value_updated(User_Post_Transition_Policy, Current_Water_Level, 
#                                                                     year, Start_Year, Year_Type[year], Predicted_Elevation)       
     
#         else:
#             ### typical policy prior to post-transition, or if user does not use post-transition policy
#             ### recall, will use the policy defined at the highest level if water level goes above that level
            
#             export_allowed_for_year = find_policy_value_updated(Export_Policy, Current_Water_Level, 
#                                                                 year, Start_Year, Year_Type[year], Predicted_Elevation, Policy_Name)


        ############################################## Apply POLICY ##########################################
        
        if (Reached_6391 and Post_Transition_Policy_to_Use != 'None'):
            
            export_allowed_for_year = find_policy_value_updated(Post_Transition_Policies, Current_Water_Level, 
                                                                year, Start_Year, Year_Type[year], Predicted_Elevation,Post_Transition_Policy_to_Use)
     
        else:
            ### typical policy prior to post-transition, or if user does not use post-transition policy
            ### recall, will use the policy defined at the highest level if water level goes above that level
            
            export_allowed_for_year = find_policy_value_updated(Export_Policy, Current_Water_Level, 
                                                                year, Start_Year, Year_Type[year], Predicted_Elevation, Policy_Name)
            
        ## make sure flow requirements are met
        flow_requirement_for_year = Amended_License_Flow_Requirement[year]

        ### in this case all flow should go towards meeting flow requirement (since grant lake release < SEF)
        if Grant_Spill_and_Outflow_for_year <= flow_requirement_for_year:
            export_allowed_for_year = 0
            Rush_Flow_from_Grant_for_year = Grant_Spill_and_Outflow_for_year

        ### simple case that is same as assuming flows are met where grant release is sufficient for exports and flow requirements
        ### note, this is the only case where exports_allowed_for_year are actually exported (since otherwise, depends on grant releases and flow requirements)
        ### note if in PT and export allowed = 99999 then this will not happen!
        elif Grant_Spill_and_Outflow_for_year >= (flow_requirement_for_year + export_allowed_for_year):
            Rush_Flow_from_Grant_for_year = Grant_Spill_and_Outflow_for_year - export_allowed_for_year

        ### somewhere in between
        ### figure out how much can export and meet flow requirement
        ### this logic will be what is applied for D-1631 Post Transition Policy (where all exports from grant release are allowed that exceed flow requirement)
        else:
            left_over_water = Grant_Spill_and_Outflow_for_year - flow_requirement_for_year
            export_allowed_for_year = left_over_water
            Rush_Flow_from_Grant_for_year = flow_requirement_for_year
        
        export_allowed_for_year = float(export_allowed_for_year)
        Rush_Flow_from_Grant_for_year = float(Rush_Flow_from_Grant_for_year)
        
        Exports_list.append(export_allowed_for_year)
        Rush_into_Mono_list.append(Rush_Flow_from_Grant_for_year)
        
        
        ############################################## Apply POLICY ##########################################

        ############ OTHER NON-MANAGED DATA #############

        ###### Ungauged Runoff for Year
        Ungauged_Runoff_for_year = Ungauged_Runoff[year]
        
        ### assuming ungauged runoff is updated based on surface area of mono lake
        Ungauged_Runoff_for_year_ft = Ungauged_Runoff_for_year
        ### multiply ft by surface area of ungauged region, which depends on mono lake current surface area
        Current_Ungauged_SA = Ungauged_Assumed_Area + (Mono_SHP_Area - Current_Surface_Area)
        Ungauged_Runoff_for_year = Ungauged_Runoff_for_year_ft * Current_Ungauged_SA

        ### Precipitation on Lake
        P_inches_for_year = Mono_Lake_P[year]  
        # convert to ft and multiply by the surface area (acres) of the lake to get ac-ft
        P_Lake_for_year = (P_inches_for_year / 12) * Current_Surface_Area  ### inches to ft (divide by 12)

        ### Evaporation from Lake
        E_inches_for_year = Mono_Lake_PET[year]
        E_Lake_for_year = (E_inches_for_year / 12) * Current_Surface_Area  ### inches to ft (divide by 12)

        ############################## SALINITY PET MODIFICATION ##################################

        ## linear-relationship for Salinity-Elevation
        salinity = Current_Water_Level*m_salinity + b_salinity
        if salinity > max_salinity_value:
            ### cannot have increased evaporation due to reduced salinity!
            salinity = max_salinity_value
        if salinity < min_salinity_value:
            ### also, cap the impact salinity can have
            salinity = min_salinity_value
        E_Lake_for_year = E_Lake_for_year*salinity

        ############################## CALCULATE STORAGE CHANGE ######################################

        ######################## from here onward, is where change in storage is recalculated
        ######################## based on average surface area of Mono Lake for P and E calculation
        ######################## making this iterative (rather than a one time calc) refines result
        
        start_of_year_Storage = float(Current_Storage)  ### need this to recaulculate current_storage
        start_of_year_Surface_Area = float(Current_Surface_Area)
        iter_surface_area = 0
        while iter_surface_area < 5:

            ### in each iteration, the storage delta and end of year (current_storage) will be refined 
            ### by getting a better estimate of the average lake level during the year
            Storage_delta = measured_inflow_for_year + Rush_Flow_from_Grant_for_year + Ungauged_Runoff_for_year + P_Lake_for_year - E_Lake_for_year - GW_Export
            
            ### Quadratic Error
            #Storage_delta = (second_coeff_error_using_observed_inflow * (Storage_delta**2)) + slope_error_using_observed_inflow * Storage_delta + intercept_error_using_observed_inflow
            ### Linear Error
            Storage_delta = slope_error_using_observed_inflow * Storage_delta + intercept_error_using_observed_inflow
            
            Current_Storage = start_of_year_Storage + Storage_delta

            ### get average surface area between current and next year to adjust P and E
            Current_Surface_Area = polyfit_func(Current_Storage, storage_obs_min, storage_obs_max, sa_obs_min, sa_obs_max, obs_storage_sa_slope_start, obs_storage_sa_slope_end, 0, obs_polyfit_storage_sa)  
            ### average surface area between current and next year
            avg_surface_area_for_current_year = (start_of_year_Surface_Area + Current_Surface_Area) / 2
            
            #print(avg_surface_area_for_current_year)

            ### update P and E at Lake based on average surface area
            P_Lake_for_year = (P_inches_for_year / 12) * avg_surface_area_for_current_year
            E_Lake_for_year = (E_inches_for_year / 12) * avg_surface_area_for_current_year

            ### also update ungauged surface area
            Current_Ungauged_SA = Ungauged_Assumed_Area + (Mono_SHP_Area - avg_surface_area_for_current_year)
            Ungauged_Runoff_for_year = Ungauged_Runoff_for_year_ft * Current_Ungauged_SA

            ############################## SALINITY PET MODIFICATION ##################################
            salinity = Current_Water_Level*m_salinity + b_salinity
            if salinity > max_salinity_value:
                ### cannot have increased evaporation due to reduced salinity!
                salinity = max_salinity_value
            if salinity < min_salinity_value:
                ### also, cap the impact salinity can have
                salinity = min_salinity_value
            E_Lake_for_year = E_Lake_for_year*salinity
            ############################## SALINITY PET MODIFICATION ##################################
            
            iter_surface_area += 1
        
        ### actually update start of years storage
        Current_Storage = start_of_year_Storage + Storage_delta
        Current_Surface_Area = float(polyfit_func(Current_Storage, storage_obs_min, storage_obs_max, sa_obs_min, sa_obs_max, obs_storage_sa_slope_start, obs_storage_sa_slope_end, 0, obs_polyfit_storage_sa))
        Current_Water_Level = float(polyfit_func(Current_Storage, storage_obs_min, storage_obs_max, elev_obs_min, elev_obs_max, obs_storage_elev_slope_start, obs_storage_elev_slope_end, min_mono_lake_elevation, obs_polyfit_storage_elev))
        
        ### add to list that will be used to compare to observed water level
        Predicted_Elevation.append(Current_Water_Level)  
        Predicted_Storage.append(Current_Storage)
        
        if (Current_Water_Level >= 6391) & (Reached_6391 == False):
            Reached_6391 = True
            Transition_Time = year
            
    ### if 6391 never reached then add transition time
    if Reached_6391 == False:
        Transition_Time = 'NaN'
    
    return Predicted_Elevation, Exports_list, Rush_into_Mono_list, Predicted_Storage


# Define a function to get policy outputs
def get_policy_output(state,
                      Model_file, 
                      Lake_file, 
                      ERA5_Data,
                      GCM_Data_Path,
                      Policy_Data, 
                      Post_Transition_Policies,
                      GCM_Member_List,
                      simple_GCM_Name_List,
                      ):
    
    key_word = state["key_word"]
    Initial_Water_Level = state["Initial_Water_Level"]
    Wrapped_or_Projections = state["Wrapped_or_Projections"]
    Policy_list = state["Pre_Transition_Policies"]
    Post_Transition_Policy_List = state["Post_Transition_Policies"]
    SSPs_of_Interest = state["SSPs_of_Interest"]
    Start_Year = state["Start_Year"]
    End_Year = state["End_Year"]

    total_policy_combos_running = len(Policy_list)*len(Post_Transition_Policy_List)
    if Wrapped_or_Projections == 'Wrapped':
        run_time_each_policy = 0.1
    else:
        run_time_each_policy = 0.05
    
    ### assuming 3 seconds per policy for GCMs, 6 seconds for wrapped runs
    estimate_of_run_time = round(total_policy_combos_running * run_time_each_policy,1)  
    print(f'For {total_policy_combos_running} Policy Combinations, Will Require Roughly {estimate_of_run_time} Minutes\n')
    
    if Wrapped_or_Projections == 'Wrapped':
        print('Wrapped Runs')
    else:
        print('Projections')
    
    num_pre_transition_polices = len(Policy_list)
    num_post_transition_policies = len(Post_Transition_Policy_List)
    print(f'Running {num_pre_transition_polices} Pre-Transition Policies with {num_post_transition_policies} Post-Transition Policies')
    
    # Load observed data range of elev-vol-sa
    mono_lake_storage_elev = pd.read_csv(Lake_file,delimiter='\t')

    # Get fit between elevation and storage/surface area
    df = mono_lake_storage_elev.copy()
    df = df.sort_values('elev_ft')  ### also sorts volume

    # Fit a quadratic polynomial to the data
    obs_polyfit_elev_storage = np.polyfit(df['elev_ft'], df['volume_ac_ft'], 2)
    # Get coefficients so can apply model
    obs_polyfit_elev_storage = np.poly1d(obs_polyfit_elev_storage)

    # Calculate the slopes based on the first two and last two values of the observed data
    obs_elev_storage_slope_start = (df['volume_ac_ft'].iloc[1] - df['volume_ac_ft'].iloc[0]) / (df['elev_ft'].iloc[1] - df['elev_ft'].iloc[0])
    obs_elev_storage_slope_end = (df['volume_ac_ft'].iloc[-1] - df['volume_ac_ft'].iloc[-2]) / (df['elev_ft'].iloc[-1] - df['elev_ft'].iloc[-2])

    # Use intercept at y_min, y_max
    storage_obs_min = np.min(df['volume_ac_ft'])
    storage_obs_max = np.max(df['volume_ac_ft'])

    elev_obs_min = np.min(df['elev_ft'])
    elev_obs_max = np.max(df['elev_ft'])
    
    ### Wrapped Runs Output sequence, policy, PT_policy, year 
    ### where sequence is a start year from 1971 to 2019 and year is the number of years into the future (typically 0 to 49)
    ### Outputs Water Level, Export, Trans_Time (SRF/SEF and RYT can be obtained from inputs or added to outputs)

    ### Projections Output: policy, PT_policy, ssp, GCM, year
    ### where year will be 77 years if going from 2024 to 2100, and GCMs are like the sequences in wrapped runs
    if Wrapped_or_Projections == 'Wrapped':

        #### Wrapped Runs
        if Start_Year < 1955:
            Start_Year = 1955
        if End_Year > 2020:
            End_Year = 2020

        wrapped_years = (Start_Year,End_Year)  ## only +1 since last year only uses the data of the last year
        Num_Years = (End_Year - Start_Year) + 2 ## +2 since uses last year data and includes following years water level prediction
        wrapped_years_list = list(np.arange(wrapped_years[0], wrapped_years[1]+1))

        SEF_or_SRF_to_Predict_Grant_Outflow = 'SEF'
        SEF_or_SRF_for_Post_Transition_Flow_Requirement = 'SEF'

        hist_era5_df = pd.read_csv(f'{ERA5_Data}/ERA5_{SEF_or_SRF_to_Predict_Grant_Outflow}_Predicted_Flow_into_Mono_Lake.csv',index_col=0)

        ### focusing on 1971-2019
        hist_era5_df = hist_era5_df[(hist_era5_df.index>=wrapped_years[0]) & (hist_era5_df.index<=wrapped_years[1])]

        ### store results for each wrapped run and policy here
        Predicted_Water_Levels_List = []
        Predicted_Exports_List = []

        for wrapped_start_yr in list(np.arange(wrapped_years[0],wrapped_years[1]+1)):

            first_years_of_wrapped_run = hist_era5_df[hist_era5_df.index>=wrapped_start_yr]
            last_years_of_wrapped_run = hist_era5_df[hist_era5_df.index<wrapped_start_yr]
            ### concatenate wrapped run data
            tmp_df = pd.concat([first_years_of_wrapped_run,last_years_of_wrapped_run])

            ### if choose to use integers from 0 to 50 (or whatever range from end - start year is)
            #tmp_df.index = np.arange(0,len(tmp_df))

            ### start of each sequence will vary, but index as start_year to end_year
            ### for consistency with checking policy export limit (for policies that are dynamic and keep track of years since start)
            tmp_df.index = np.arange(Start_Year,End_Year+1)

            ### actual flow
            measured_Lee_Vining = tmp_df['Lee Vining']
            measured_Walker_Parker = tmp_df['Walker+Parker']
            measured_inflow_not_managed_by_DWP = tmp_df['Basins Not Managed By DWP']
            measured_Grant_Lake_Spillway_Outflow = tmp_df['Grant Lake Spillway + Outflow']
            Ungauged_Runoff = tmp_df['Ungauged']  ### BCd in water level notebook based on optimized ungauged modification
            Mono_Lake_P = tmp_df['Mono_Lake_P']
            Mono_Lake_PET = tmp_df['Mono_Lake_PET']
            Year_Type = tmp_df['RYT']
            Amended_License_Flow_Requirement = tmp_df[f'{SEF_or_SRF_for_Post_Transition_Flow_Requirement} Rush (ac-ft/year)']

            for post_transition_policy in Post_Transition_Policy_List:

                for Policy in Policy_list:
                    
                    #print(f'Policy: {Policy} + {post_transition_policy}')

                    Predicted_Elevation, Exports_list, Rush_into_Mono_list, Predicted_Storage = predict_Mono_Lake_Water_Level_Added_Policies(
                        Model_file,
                        Lake_file,
                        measured_Lee_Vining,
                        measured_Walker_Parker,
                        measured_inflow_not_managed_by_DWP,                                                                                                 
                        measured_Grant_Lake_Spillway_Outflow,
                        Mono_Lake_P,                                                                                                                     
                        Mono_Lake_PET,                                                                                                                     
                        Ungauged_Runoff,                                                                                                                     
                        Year_Type,                                                                                                                      
                        Amended_License_Flow_Requirement,
                        Initial_Water_Level,
                        Start_Year, End_Year,
                        Policy_Data,              ### Pre-Transition Policy Dataset
                        Post_Transition_Policies, ### Post-Transition Policy Dataset...UPDATED THIS SO LOOPS THROUGH POST-TRANSITION POLICIES
                        Policy_Name = Policy,                   ### Used to index the Pre-Transition Policy from the Policy_Data Dataset                                                             
                        Post_Transition_Policy_to_Use=post_transition_policy    ### Used to index the Post-Transition Policy from the Post_Transition_Policies Dataset          
                        )

                    Exports_list.append(0)  ### add zero to end since only estimate water levels/storage for last time-stamp
                    Predicted_Water_Levels_List.append((wrapped_start_yr, Policy, post_transition_policy, Predicted_Elevation))
                    Predicted_Exports_List.append((wrapped_start_yr, Policy, post_transition_policy, Exports_list))


        # Convert lists to arrays
        num_policies = len(Policy_list)
        num_PT_policies = len(Post_Transition_Policy_List)

        # Initialize empty arrays
        Predicted_Water_Levels_Array = np.empty((len(wrapped_years_list), num_policies, num_PT_policies, Num_Years))
        Predicted_Exports_Array = np.empty((len(wrapped_years_list), num_policies, num_PT_policies, Num_Years))

        # Fill arrays with data
        for i, (wrapped_start_yr, Policy, post_transition_policy, predictions) in enumerate(Predicted_Water_Levels_List):
            wrapped_index = wrapped_years_list.index(wrapped_start_yr)
            policy_index = Policy_list.index(Policy)
            post_transition_index = Post_Transition_Policy_List.index(post_transition_policy)
            Predicted_Water_Levels_Array[wrapped_index, policy_index, post_transition_index, :] = predictions

        for i, (wrapped_start_yr, Policy, post_transition_policy, predictions) in enumerate(Predicted_Exports_List):
            wrapped_index = wrapped_years_list.index(wrapped_start_yr)
            policy_index = Policy_list.index(Policy)
            post_transition_index = Post_Transition_Policy_List.index(post_transition_policy)
            Predicted_Exports_Array[wrapped_index, policy_index, post_transition_index, :] = predictions

        # Create xarray Dataset
        Policy_Outputs = xr.Dataset(data_vars=dict(
            Water_Level=(["Sequence", "policy", "PT_policy", "year"], Predicted_Water_Levels_Array)))

        # Apply the polyfit_elev_storage function to the Water_Level variable
        #Storage_array = polyfit_elev_storage(Policy_Outputs["Water_Level"])
        Storage_array = polyfit_func(Policy_Outputs["Water_Level"], elev_obs_min, elev_obs_max, storage_obs_min, storage_obs_max, 
                          obs_elev_storage_slope_start, obs_elev_storage_slope_end, 0, obs_polyfit_elev_storage)
        # Add the Storage array as a new variable in the Policy_Outputs dataset
        Policy_Outputs = Policy_Outputs.assign(Storage=(["Sequence", "policy","PT_policy", "year"], Storage_array))

        # Add Exports
        Policy_Outputs = Policy_Outputs.assign(Exports=(["Sequence", "policy","PT_policy", "year"], Predicted_Exports_Array))

        ### Add coordinate information (and transition time as variable

        ### follows order of for loops
        Policy_Outputs = Policy_Outputs.assign_coords(Sequence=wrapped_years_list)
        Policy_Outputs = Policy_Outputs.assign_coords(policy=Policy_list)
        Policy_Outputs = Policy_Outputs.assign_coords(PT_policy=Post_Transition_Policy_List)
        Policy_Outputs = Policy_Outputs.assign_coords(year=range(0,Num_Years))

        Trans_Time = (Policy_Outputs["Water_Level"] >= 6391).argmax(dim="year")

        ### HARD-CODED YEARS
        Trans_Time = Trans_Time.where(Trans_Time!=0, Num_Years)

        Policy_Outputs = Policy_Outputs.assign(Transition_Time = Trans_Time)

        print('Wrapped Done')


    if Wrapped_or_Projections == 'Projections':

        ## Outputs ##
        # xarray with coordinates...
        # post transition policy: 
        # ssp: ["ssp245","ssp370","ssp585"]
        # GCM: ['ACCESS-CM2','CNRM-ESM2-1','EC-Earth3-Veg','FGOALS-g3','GFDL-ESM4','INM-CM5-0','IPSL-CM6A-LR',
        #       'KACE-1-0-G','MIROC6','MPI-ESM1-2-HR','MRI-ESM2-0']
        # policy : ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']
        # year : 2024-2100
        # with variables...
        # Water Level : Lake water level (ft)
        # Exports : Water Exports (af)
        # Trans_Time: Transition Time to 6391 ft (years)

        ### For now keeping this hard-coded (can later decide if want users to modify)
        Historical_or_Dynamic_RYT = 'Dynamic'    ### will typically keep dynamic (updates RYT definition based on GCM)
        SEF_or_SRF_to_Predict_Grant_Outflow = 'SEF'
        SEF_or_SRF_for_Post_Transition_Flow_Requirement = 'SEF'

        ### used as x-axis in plots
        ### used for indexing results...initial condition = start_year, last year of full runoff year = 2099, 
        ### which is used to predict 2100 water level (hence end_year + 2)
        GCM_Range_of_Years = np.arange(Start_Year,End_Year+2)  
        ### used to define transition time if does not occur
        Num_Years = (End_Year - Start_Year)+2 ### +2 rather than 1, since 2099 represents the 2099 runoff year which includes a 2100 water level estimate

        # Initialize storage for predicted values
        Predicted_Water_Levels = []
        Predicted_Exports = []

        # Loop through policies
        for Policy in Policy_list:
            Predicted_Water_Levels_Trans = []
            Predicted_Exports_Trans = []

            for post_transition_policy in Post_Transition_Policy_List:
                Predicted_Water_Levels_ssp = []
                Predicted_Exports_ssp = []
                
                print(f'Policy: {Policy} + {post_transition_policy}')

                for ssp in SSPs_of_Interest:
                    Predicted_Water_Levels_GCM = []
                    Predicted_Exports_GCM = []

                    for gcm in GCM_Member_List:
                        tmp_df = pd.read_csv(f'{GCM_Data_Path}/{Historical_or_Dynamic_RYT}_RYT_{SEF_or_SRF_to_Predict_Grant_Outflow}/{gcm}_{ssp}.csv', index_col=0)

                        # Extract relevant columns
                        measured_Lee_Vining = tmp_df['Lee Vining']
                        measured_Walker_Parker = tmp_df['Walker+Parker']
                        measured_inflow_not_managed_by_DWP = tmp_df['Basins Not Managed By DWP']
                        measured_Grant_Lake_Spillway_Outflow = tmp_df['Grant Lake Spillway + Outflow']
                        Ungauged_Runoff = tmp_df['Ungauged']
                        Mono_Lake_P = tmp_df['Mono_Lake_P']
                        Mono_Lake_PET = tmp_df['Mono_Lake_PET']
                        Year_Type = tmp_df['RYT']
                        Amended_License_Flow_Requirement = tmp_df[f'{SEF_or_SRF_for_Post_Transition_Flow_Requirement} Rush (ac-ft/year)']

                        # Predict water levels and exports
                        Predicted_Elevation, Exports_list, Rush_into_Mono_list, Predicted_Storage = predict_Mono_Lake_Water_Level_Added_Policies(
                            ######### FILE LOCATIONS ##########
                            Model_file,
                            Lake_file,
                            measured_Lee_Vining,
                            measured_Walker_Parker,
                            measured_inflow_not_managed_by_DWP,
                            measured_Grant_Lake_Spillway_Outflow,
                            Mono_Lake_P,                                                                                                                     
                            Mono_Lake_PET,
                            Ungauged_Runoff,
                            Year_Type,                                                                                                                      
                            Amended_License_Flow_Requirement,
                            ##### THESE ARE FLEXIBLE USER-DEFINED OPTIONS ###
                            Initial_Water_Level,
                            Start_Year, End_Year,
                            Policy_Data,              ### Pre-Transition Policy Dataset
                            Post_Transition_Policies, ### Post-Transition Policy Dataset...UPDATED THIS SO LOOPS THROUGH POST-TRANSITION POLICIES
                            Policy_Name = Policy,                   ### Used to index the Pre-Transition Policy from the Policy_Data Dataset                                                             
                            Post_Transition_Policy_to_Use=post_transition_policy    ### Used to index the Post-Transition Policy from the Post_Transition_Policies Dataset          
                        )
                        
                        # Append results to GCM list
                        Exports_list.append(0)  # Setting Exports for Last Year to Zero since Only Get Water Level for Last Year
                        Predicted_Water_Levels_GCM.append(Predicted_Elevation)
                        Predicted_Exports_GCM.append(Exports_list)

                    # Append GCM results to SSP list
                    Predicted_Water_Levels_ssp.append(Predicted_Water_Levels_GCM)
                    Predicted_Exports_ssp.append(Predicted_Exports_GCM)

                # Append SSP results to post-transition policy list
                Predicted_Water_Levels_Trans.append(Predicted_Water_Levels_ssp)
                Predicted_Exports_Trans.append(Predicted_Exports_ssp)

            # Append post-transition policy results to policy list
            Predicted_Water_Levels.append(Predicted_Water_Levels_Trans)
            Predicted_Exports.append(Predicted_Exports_Trans)

        # Define year list for coordinates
        year_list = list(np.arange(Start_Year, End_Year + 2))

        # Create xarray Dataset
        Policy_Outputs = xr.Dataset(data_vars=dict(
            Water_Level=(["policy", "PT_policy", "ssp", "GCM", "year"], Predicted_Water_Levels)
        ))

        # Apply the polyfit_elev_storage function to the Water_Level variable
        #Storage_array = polyfit_elev_storage(Policy_Outputs["Water_Level"])
        Storage_array = polyfit_func(Policy_Outputs["Water_Level"], elev_obs_min, elev_obs_max, storage_obs_min, storage_obs_max, 
                          obs_elev_storage_slope_start, obs_elev_storage_slope_end, 0, obs_polyfit_elev_storage)
        # Add the Storage array as a new variable in the Policy_Outputs dataset
        Policy_Outputs = Policy_Outputs.assign(Storage=(["policy", "PT_policy", "ssp", "GCM", "year"], Storage_array))

        # Add Exports
        Policy_Outputs = Policy_Outputs.assign(Exports=(["policy", "PT_policy", "ssp", "GCM", "year"], Predicted_Exports))

        # Assign coordinates
        Policy_Outputs = Policy_Outputs.assign_coords(policy=Policy_list)
        Policy_Outputs = Policy_Outputs.assign_coords(PT_policy=Post_Transition_Policy_List)
        Policy_Outputs = Policy_Outputs.assign_coords(ssp=SSPs_of_Interest)
        Policy_Outputs = Policy_Outputs.assign_coords(GCM=simple_GCM_Name_List)
        Policy_Outputs = Policy_Outputs.assign_coords(year=year_list)

        # Calculate transition time
        Trans_Time = (Policy_Outputs["Water_Level"] >= 6391).argmax(dim="year")

        # Handle edge case where transition doesn't occur
        Trans_Time = Trans_Time.where(Trans_Time != 0, Num_Years)

        # Assign transition time to Dataset
        Policy_Outputs = Policy_Outputs.assign(Transition_Time=Trans_Time)

        print('Projections done!')
        
    return Policy_Outputs
    
def plot_variable_time_series_by_model(Wrapped_or_Projections, Policy_of_Interest, PT_Policy_of_Interest, Variable, SSP_of_Interest, Policy_Outputs,key_word,Start_Year,End_Year):
    
    #key_word, Start_Year, End_Year = set_attributes(Wrapped_or_Projections)
    x_years = list(Policy_Outputs['year'].values)

    tmp_unit = dict_of_variable_to_unit[Variable]
    variable_label = Variable.replace('_', ' ') if '_' in Variable else Variable

    ### Plot every sequence or GCM+SSP for given policy (use single color for each sequence/GCM+SSP)
    if Wrapped_or_Projections == 'Wrapped':
        tmp_array = Policy_Outputs.sel(policy=Policy_of_Interest,PT_policy=PT_Policy_of_Interest)
    elif Wrapped_or_Projections == 'Projections':
        tmp_array = Policy_Outputs.sel(policy=Policy_of_Interest,PT_policy=PT_Policy_of_Interest,ssp=SSP_of_Interest)

    # Plot Water_Level for each GCM
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over each GCM and plot the Water_Level
    for clim in tmp_array[key_word].values:
        if Wrapped_or_Projections == 'Wrapped':
            tmp_array.sel(Sequence=clim)[Variable].plot.line(ax=ax)
        elif Wrapped_or_Projections == 'Projections':
            tmp_array.sel(GCM=clim)[Variable].plot.line(ax=ax, label=clim)

    # Customize the plot
    ax.set_title(f"{variable_label} for Different {key_word}'s ({Start_Year}-{End_Year+1})\nPolicy: {Policy_of_Interest}, PT-Policy: {PT_Policy_of_Interest}")
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{variable_label} ({tmp_unit})')

    if Wrapped_or_Projections == 'Projections':
        ax.legend(title=key_word, bbox_to_anchor=(1.05, 1), loc='upper left',title_fontsize=16)

    if Variable == 'Water_Level':
        ax.axhline(6392,color='black',linestyle='--')
        ax.axhline(6391,color='black',linestyle='--')
        ax.axhline(6388,color='black',linestyle='--')
        ax.axhline(6386,color='black',linestyle='--')
        
        ax.text(End_Year,6392+0.25,'6392',color='grey',fontsize=11)
        ax.text(End_Year,6391+0.25,'6391',color='grey',fontsize=11)
        ax.text(End_Year,6388+0.25,'6388',color='grey',fontsize=11)
        ax.text(End_Year,6386+0.25,'6386',color='grey',fontsize=11)
        
def plot_variable_time_series_by_policy(Wrapped_or_Projections, policies, pt_policies_to_plot, Variable, SSP_of_Interest, Policy_Outputs,key_word,Start_Year,End_Year,Sequence_of_Interest):
    
    clim = Sequence_of_Interest
    
    #key_word, Start_Year, End_Year = set_attributes(Wrapped_or_Projections)
    #x_years = list(Policy_Outputs['year'].values)
    
    tmp_unit = dict_of_variable_to_unit[Variable]
    variable_label = Variable.replace('_', ' ') if '_' in Variable else Variable

    # sequences or GCMs that will loop through
    clims = list(Policy_Outputs[key_word].values)

    #pt_policies = list(Policy_Outputs.PT_policy.values)

    if Wrapped_or_Projections == 'Projections':
        tmp_array = Policy_Outputs.sel(ssp=SSP_of_Interest)
    elif Wrapped_or_Projections == 'Wrapped':
        tmp_array = Policy_Outputs.copy(deep=True)

    # # Iterate over each sequence or GCM and plot every policy combo
    # for clim in Sequences:

    fig, ax = plt.subplots(figsize=(10, 6))

    for tmp_policy in policies:

        for tmp_pt_policy in pt_policies_to_plot:

            if len(pt_policies_to_plot) == 1:
                tmp_pt_policy_label = ''
                tmp_pt_title_label = f'Post-Transition: {tmp_pt_policy}'
            else:
                tmp_pt_policy_label = f', PT: {tmp_pt_policy}'
                tmp_pt_title_label = ''

            if 'U' in tmp_policy:
                linestyle = '--'
            else:
                linestyle = '-'

            if Wrapped_or_Projections == 'Wrapped':
                tmp_array.sel(policy=tmp_policy,PT_policy=tmp_pt_policy,Sequence=Sequence_of_Interest)[Variable].plot.line(ax=ax,label=f'{tmp_policy}{tmp_pt_policy_label}',linestyle=linestyle)
            elif Wrapped_or_Projections == 'Projections':
                tmp_array.sel(policy=tmp_policy,PT_policy=tmp_pt_policy,GCM=Sequence_of_Interest)[Variable].plot.line(ax=ax,label=f'{tmp_policy}{tmp_pt_policy_label}',linestyle=linestyle)

            # Customize the plot
            ax.set_title(f"{variable_label} for {key_word} {clim} ({Start_Year}-{End_Year+1})\n{tmp_pt_title_label}")
            ax.set_xlabel('Year')
            ax.set_ylabel(f'{variable_label} ({tmp_unit})')

            ax.legend(title='Policy', bbox_to_anchor=(1.05, 1), loc='upper left',title_fontsize=16)

            if Variable == 'Water_Level':
                ax.axhline(6392,color='black',linestyle='--')
                ax.axhline(6391,color='black',linestyle='--')
                ax.axhline(6388,color='black',linestyle='--')
                ax.axhline(6386,color='black',linestyle='--')

                ax.text(End_Year,6392+0.25,'6392',color='grey',fontsize=11)
                ax.text(End_Year,6391+0.25,'6391',color='grey',fontsize=11)
                ax.text(End_Year,6388+0.25,'6388',color='grey',fontsize=11)
                ax.text(End_Year,6386+0.25,'6386',color='grey',fontsize=11)
                #ax.axhline(6384,color='black',linestyle='--')

def plot_variable_time_series_by_policy_mean(Wrapped_or_Projections, policies, pt_policies_to_plot, Variable, SSP_of_Interest, Policy_Outputs,key_word, Start_Year, End_Year, Stdev=False, Min_Max=False):
    
    #key_word, Start_Year, End_Year = set_attributes(Wrapped_or_Projections)
    x_years = list(Policy_Outputs['year'].values)
    
    tmp_unit = dict_of_variable_to_unit[Variable]
    variable_label = Variable.replace('_', ' ') if '_' in Variable else Variable

    # sequences or GCMs that will loop through
    clims = list(Policy_Outputs[key_word].values)
    #pt_policies = list(Policy_Outputs.PT_policy.values)

    if Wrapped_or_Projections == 'Projections':
        tmp_array = Policy_Outputs.sel(ssp=SSP_of_Interest)
    elif Wrapped_or_Projections == 'Wrapped':
        tmp_array = Policy_Outputs.copy(deep=True)

    # Iterate over each sequence or GCM and plot every policy combo

    # Define a color cycle (optional: you can customize this list of colors)
    colors = plt.cm.tab10.colors  # This uses the 'tab10' colormap, you can choose any color map
    color_cycle = itertools.cycle(colors)  # Create an iterator that cycles through the colors

    # Create the figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over each sequence or GCM and plot every policy combo
    for tmp_policy in policies:

        for tmp_pt_policy in pt_policies_to_plot:

            tmp_tmp_array = tmp_array.sel(policy=tmp_policy, PT_policy=tmp_pt_policy)[Variable]

            tmp_mean = tmp_tmp_array.mean(dim=key_word)
            tmp_std = tmp_tmp_array.std(dim=key_word)
            tmp_min = tmp_tmp_array.min(dim=key_word)
            tmp_max = tmp_tmp_array.max(dim=key_word)

            # Get the next color from the manual color cycle
            color = next(color_cycle)

            if len(pt_policies_to_plot) == 1:
                tmp_pt_policy_label = ''
            else:
                tmp_pt_policy_label = f' & {tmp_pt_policy}'

            if 'U' in tmp_policy:
                linestyle = '--'
            else:
                linestyle = '-'

            # Plot the mean using the current color
            ax.plot(x_years, tmp_mean, label=f'{tmp_policy}{tmp_pt_policy_label}', color=color,linestyle=linestyle)

            if Stdev:
                # Plot the standard deviation as a shaded area using the current color
                ax.fill_between(x_years, tmp_mean - tmp_std, tmp_mean + tmp_std, color=color, alpha=0.3)

            if Min_Max:
                # Plot the minimum and maximum as dashed lines using the current color
                ax.plot(x_years, tmp_min, linestyle=':', color=color)
                ax.plot(x_years, tmp_max, linestyle=':', color=color)

    # Customize the plot
    if len(pt_policies_to_plot) == 1:
        title_pt_policy_label = f' and Post-Transition: {tmp_pt_policy}'
    else:
        title_pt_policy_label = ''

    if Wrapped_or_Projections == 'Wrapped':
        ax.set_title(f"{variable_label} for Different Policies{title_pt_policy_label}\nWrapped Runs: {Start_Year}-{End_Year+1}")
    elif Wrapped_or_Projections == 'Projections':
        ax.set_title(f"{variable_label} for Different Policies{title_pt_policy_label}\nProjections: {Start_Year}-{End_Year+1}")

    ax.set_xlabel('Year')
    ax.set_ylabel(f'{variable_label} ({tmp_unit})')

    # Add legend and grid
    ax.legend(title='Policy', bbox_to_anchor=(1.05, 1), loc='upper left', title_fontsize=16)

    if Variable == 'Water_Level':
        ax.axhline(6392,color='black',linestyle='--')
        ax.axhline(6391,color='black',linestyle='--')
        ax.axhline(6388,color='black',linestyle='--')
        ax.axhline(6386,color='black',linestyle='--')
        #ax.axhline(6384,color='black',linestyle='--')

        ax.text(End_Year,6392+0.15,'6392',color='black',fontsize=11)
        ax.text(End_Year,6391+0.15,'6391',color='black',fontsize=11)
        ax.text(End_Year,6388+0.15,'6388',color='black',fontsize=11)
        ax.text(End_Year,6386+0.15,'6386',color='black',fontsize=11)

### Boxplot for different policies performance (can choose any variable to plot from output)
def boxplot_variable_by_policy(Wrapped_or_Projections, policies, pt_policies_to_plot, Variable, year_of_interest, SSP_of_Interest, Policy_Outputs):
    
    tmp_unit = dict_of_variable_to_unit[Variable]
    variable_label = Variable.replace('_', ' ') if '_' in Variable else Variable

    # Select the appropriate data based on Wrapped or Projections
    if Wrapped_or_Projections == 'Projections':
        tmp_array = Policy_Outputs.sel(ssp=SSP_of_Interest)
    elif Wrapped_or_Projections == 'Wrapped':
        tmp_array = Policy_Outputs.copy(deep=True)

    # Create the figure and axis for the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Initialize an empty list to hold the boxplot positions
    boxplot_positions = []
    boxplot_labels = []

    # Define the width of each boxplot and the space between them
    box_width = 0.6
    space_between = 1.0
    current_position = 1.0

    # Iterate over each policy and PT_policy combination
    for tmp_policy in policies:
        for tmp_pt_policy in pt_policies_to_plot:

            tmp_tmp_array = tmp_array.sel(policy=tmp_policy, PT_policy=tmp_pt_policy)[Variable]

            if Variable == 'Transition_Time':
                data_to_plot = tmp_tmp_array.copy()
                data_flattened = data_to_plot.copy()
            else:
                # Select data only for the year of interest
                data_to_plot = tmp_tmp_array.sel(year=year_of_interest).values
                # Flatten the data across the policies for the boxplot
                data_flattened = data_to_plot.flatten()

            # Plot the boxplot for the selected range
            ### whis=[0, 100] sets whiskers to min/max
            ax.boxplot(data_flattened, positions=[current_position], widths=box_width, whis=[0, 100], showmeans=True, 
                       meanline=False, meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"10"})

            # Add the label for this policy combination
            if len(pt_policies_to_plot) == 1:
                boxplot_labels.append(f'{tmp_policy}')
            else:
                boxplot_labels.append(f'{tmp_policy}, PT: {tmp_pt_policy}')
            boxplot_positions.append(current_position)

            # Move to the next position
            current_position += box_width + space_between

    # Customize the plot
    if len(pt_policies_to_plot) == 1:
        if tmp_policy == 'None':
            ax.set_title(f"{variable_label} for Different Policies\nFor Year: {year_of_interest}\nPT Policy: Same as Pre-Transition")
        else:
            ax.set_title(f"{variable_label} for Different Policies\nFor Year: {year_of_interest}\nPT Policy: {tmp_pt_policy}")
    else:
        ax.set_title(f"{variable_label} for Different Policies\nFor Year: {year_of_interest}")
    ax.set_xlabel('Policy Combinations')
    ax.set_ylabel(f'{variable_label} ({tmp_unit})')
    ax.set_xticks(boxplot_positions)
    ax.set_xticklabels(boxplot_labels, rotation=45, ha='right')

    # Create custom legend elements
    median_line = Line2D([0], [0], color='orange', lw=2, label='Median')
    mean_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Mean')
    box_edge = mpatches.Patch(color='black', label='25th-75th Percentile (Box)')
    whisker_line = Line2D([0], [0], color='black', linestyle='-', lw=1, label='Min/Max (Whiskers)')

    # Add the custom legend to the plot
    ax.legend(handles=[median_line, mean_marker, box_edge, whisker_line], bbox_to_anchor=(1.025, 1), loc='upper left', fontsize=12)

    if Variable == 'Water_Level':

        # Draw dashed lines and label them
        levels = [6392, 6391, 6388, 6386]
        for level in levels:
            ax.axhline(level, color='grey', linestyle='--')
            # Add label to the right of the dashed lines
            ax.text(ax.get_xlim()[1], level, f'{level}', va='center', ha='left', color='grey', fontsize=12)

        ax.set_ylim(6370, 6420)

    plt.tight_layout()

def boxplot_variables_by_policies(Wrapped_or_Projections, policies_to_plot, pt_policies_to_plot, variables_to_plot, year_of_interest, 
                                  SSP_of_Interest, Policy_Outputs,key_word, Start_Year, End_Year):
    
    #key_word, Start_Year, End_Year = set_attributes(Wrapped_or_Projections)
    
    ### to make sure whiskers are included by default in later plots
    include_whiskers = True
    whiskerprops = dict(color='black', linewidth=1.5)  # Show whiskers
    capprops = dict(color='black', linewidth=1.5)  # Show caps

    # Select the appropriate data based on Wrapped or Projections
    if Wrapped_or_Projections == 'Projections':
        tmp_array = Policy_Outputs.sel(ssp=SSP_of_Interest)
    elif Wrapped_or_Projections == 'Wrapped':
        tmp_array = Policy_Outputs.copy(deep=True)

    # Create the figure with dynamic rows and columns based on user input
    num_rows = len(variables_to_plot)
    num_cols = len(pt_policies_to_plot)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))

    # Iterate over each variable to determine the y-axis limits
    y_limits = {}
    for Variable in variables_to_plot:
        y_min = float('inf')
        y_max = float('-inf')

        for tmp_policy in policies_to_plot:
            for tmp_pt_policy in pt_policies_to_plot:
                tmp_tmp_array = tmp_array.sel(policy=tmp_policy, PT_policy=tmp_pt_policy)[Variable]

                if Variable == 'Exports':
                    # Calculate cumulative exports from Start_Year to year_of_interest
                    data_to_plot = tmp_tmp_array.sel(year=slice(Start_Year, year_of_interest)).sum(dim='year').values.flatten()
                else:
                    # Select data only for the year of interest
                    data_to_plot = tmp_tmp_array.sel(year=year_of_interest).values.flatten()

                y_min = min(y_min, data_to_plot.min())
                y_max = max(y_max, data_to_plot.max())

        # Calculate the range and extend by 5%
        y_range = y_max - y_min
        y_min -= 0.05 * y_range
        y_max += 0.05 * y_range

        y_limits[Variable] = (y_min, y_max)

    # Create a colormap
    colormap = cm.get_cmap('tab20', len(policies_to_plot))

    # Modify the boxplot loop to include different colors
    for row_idx, Variable in enumerate(variables_to_plot):
        tmp_unit = dict_of_variable_to_unit[Variable]

        variable_label = Variable.replace('_', ' ') if '_' in Variable else Variable

        # Unique case for this particular plot which represents cumulative exports
        if Variable == 'Exports':
            variable_label = 'Cumulative Exports'

        for col_idx, tmp_pt_policy in enumerate(pt_policies_to_plot):

            ax = axes[row_idx, col_idx] if num_rows > 1 else axes[col_idx]
            boxplot_positions = []
            boxplot_labels = []
            box_width = 0.6
            space_between = 1.0
            current_position = 1.0

            for idx, tmp_policy in enumerate(policies_to_plot):
                tmp_tmp_array = tmp_array.sel(policy=tmp_policy, PT_policy=tmp_pt_policy)[Variable]

                if Variable == 'Exports':
                    # Calculate cumulative exports from Start_Year to year_of_interest
                    data_to_plot = tmp_tmp_array.sel(year=slice(Start_Year, year_of_interest)).sum(dim='year').values
                else:
                    # Select data only for the year of interest
                    data_to_plot = tmp_tmp_array.sel(year=year_of_interest).values

                data_flattened = data_to_plot.flatten()

                # Plot the boxplot with different colors
                boxprops = dict(facecolor=colormap(idx), color='black')
                medianprops = dict(color='orange', linewidth=2)
                meanprops = dict(marker='o', markerfacecolor='red', markeredgecolor='black', markersize=10)
                if include_whiskers == False:
                    whiskerprops = dict(linestyle='None')  # Hide whiskers
                    capprops = dict(linestyle='None')  # Hide caps


                # Use patch_artist=True to fill the boxes with colors
                ax.boxplot(data_flattened, positions=[current_position], widths=box_width, showmeans=True,  #whis=[0, 100]
                           meanline=False, meanprops=meanprops, medianprops=medianprops, boxprops=boxprops,
                           whiskerprops=whiskerprops,capprops=capprops,patch_artist=True)


                boxplot_labels.append(f'{tmp_policy}')
                boxplot_positions.append(current_position)

                current_position += box_width + space_between

            # Customize the subplot
            ax.set_ylabel(f'{variable_label} ({tmp_unit})', fontsize=16)
            ax.set_xticks(boxplot_positions)
            ax.set_xticklabels(boxplot_labels, rotation=45, ha='right')
            ax.set_title(f"PT Policy: {tmp_pt_policy}")

            # Set consistent y-axis limits for the variable
            ax.set_ylim(y_limits[Variable])

            # Add Water Level specific annotations
            if Variable == 'Water_Level':
                levels = [6392, 6391, 6388, 6386, 6384]
                for level in levels:
                    ax.axhline(level, color='grey', linestyle='--')
                    ax.text(ax.get_xlim()[1], level, f'{level}', va='center', ha='left', color='grey', fontsize=12)

            else:
                ax.grid(axis='y')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Create custom legend elements
    median_line = Line2D([0], [0], color='orange', lw=2, label='Median')
    mean_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Mean')
    box_edge = mpatches.Patch(color='black', label='25th-75th Percentile (Box)')
    whisker_line = Line2D([0], [0], color='black', linestyle='-', lw=1, label='Min/Max (Whiskers)')

    # Add the custom legend to the plot
    fig.legend(handles=[median_line, mean_marker, box_edge, whisker_line], bbox_to_anchor=(1.0, 0.97), loc='upper left', fontsize=14)

    # Add an overall title for the figure
    fig.suptitle(f'Policy Performance Evaluation for Year: {year_of_interest}', fontsize=24, y=1.04)
    
def plot_variables_by_policies(Wrapped_or_Projections, policies_to_plot, pt_policies_to_plot, variables_to_plot, year_of_interest, 
                               SSP_of_Interest, Policy_Outputs, key_word, Start_Year, End_Year, use_std=True, use_median=False, include_global_limit=False):
    
    #key_word, Start_Year, End_Year = set_attributes(Wrapped_or_Projections)
    
    # Select the appropriate data based on Wrapped or Projections
    if Wrapped_or_Projections == 'Projections':
        tmp_array = Policy_Outputs.sel(ssp=SSP_of_Interest)
    elif Wrapped_or_Projections == 'Wrapped':
        tmp_array = Policy_Outputs.copy(deep=True)

    # Initialize variables for global min and max
    global_min_water_level = float('inf')
    global_max_water_level = float('-inf')
    global_min_exports = float('inf')
    global_max_exports = float('-inf')

    # Iterate over each PT policy to calculate the global min and max
    for col_idx, tmp_pt_policy in enumerate(pt_policies_to_plot):

        # Iterate over each policy's data for the current PT policy
        for idx, tmp_policy in enumerate(policies_to_plot):

            # Select the water level and export data for the current policy and PT policy
            water_level_array = tmp_array.sel(policy=tmp_policy, PT_policy=tmp_pt_policy)['Water_Level']
            export_array = tmp_array.sel(policy=tmp_policy, PT_policy=tmp_pt_policy)['Exports']

            # Calculate cumulative exports from Start_Year to year_of_interest across 11 GCMs
            cumulative_exports = export_array.sel(year=slice(Start_Year, year_of_interest)).sum(dim='year').values

            # Select water levels for the year of interest across 11 GCMs
            water_levels = water_level_array.sel(year=year_of_interest).values

            # Compute the mean and standard deviation for water levels and exports across GCMs
            mean_water_level = np.mean(water_levels)
            std_water_level = np.std(water_levels)
            mean_exports = np.mean(cumulative_exports)
            std_exports = np.std(cumulative_exports)

            # Calculate min/max based on whether to use standard deviation
            if use_std:
                min_water_level = mean_water_level - std_water_level
                max_water_level = mean_water_level + std_water_level
                min_exports = mean_exports - std_exports
                max_exports = mean_exports + std_exports
            else:
                min_water_level = mean_water_level
                max_water_level = mean_water_level
                min_exports = mean_exports
                max_exports = mean_exports

            # Update global min/max values
            global_min_water_level = min(global_min_water_level, min_water_level)
            global_max_water_level = max(global_max_water_level, max_water_level)
            global_min_exports = min(global_min_exports, min_exports)
            global_max_exports = max(global_max_exports, max_exports)

    # Create the figure with one row and multiple columns based on pt_policies_to_plot
    num_cols = len(pt_policies_to_plot)
    fig, axes = plt.subplots(1, num_cols, figsize=(8 * num_cols, 6))

    # Create a colormap
    colormap = cm.get_cmap('tab20', len(policies_to_plot))

    # Iterate over each PT policy to create the scatter plot (one PT policy per column)
    for col_idx, tmp_pt_policy in enumerate(pt_policies_to_plot):
        ax = axes[col_idx] if num_cols > 1 else axes  # Adjust for single or multiple columns

        # Plot each policy's data for the current PT policy
        for idx, tmp_policy in enumerate(policies_to_plot):

            # Select the water level and export data for the current policy and PT policy
            water_level_array = tmp_array.sel(policy=tmp_policy, PT_policy=tmp_pt_policy)['Water_Level']
            export_array = tmp_array.sel(policy=tmp_policy, PT_policy=tmp_pt_policy)['Exports']

            # Calculate cumulative exports from Start_Year to year_of_interest across 11 GCMs
            cumulative_exports = export_array.sel(year=slice(Start_Year, year_of_interest)).sum(dim='year').values

            # Select water levels for the year of interest across 11 GCMs
            water_levels = water_level_array.sel(year=year_of_interest).values

            # Compute the mean and standard deviation for water levels and exports across GCMs
            mean_water_level = np.mean(water_levels)
            std_water_level = np.std(water_levels)
            mean_exports = np.mean(cumulative_exports)
            std_exports = np.std(cumulative_exports)

            if 'U' in tmp_policy:
                marker_style = '*'
            else:
                marker_style = 'o'

            if use_std:
                # Plot the scatter plot with error bars representing 1 standard deviation
                ax.errorbar(mean_water_level, mean_exports, 
                            xerr=std_water_level, yerr=std_exports, 
                            fmt='o', color=colormap(idx), label=f'{tmp_policy}', 
                            markersize=10, alpha=0.7,marker=marker_style)
            else:
                # Plot without error bars
                ax.scatter(mean_water_level, mean_exports,
                           color=colormap(idx),
                           label=f'{tmp_policy}', 
                           s=100, alpha=0.7,marker=marker_style)

            # Annotate every other tmp_policy
            if idx % 2 == 0:  # Annotate every other policy

                if use_std == True:
                    ax.annotate(f'{tmp_policy}', 
                                xy=(mean_water_level, mean_exports), 
                                xytext=(mean_water_level + 1, mean_exports + 10000),  # Adjust offsets as needed
                                arrowprops=dict(facecolor='black', edgecolor='black', 
                                                shrink=0.05, width=0.5, headwidth=7, headlength=7, 
                                                alpha=0.7),
                                fontsize=12, fontweight='bold', color=colormap(idx))
                else:
                    ax.annotate(f'{tmp_policy}', 
                                xy=(mean_water_level, mean_exports), 
                                xytext=(mean_water_level + 0.25, mean_exports + 500),  # Adjust offsets as needed
                                arrowprops=dict(facecolor='black', edgecolor='black', 
                                                shrink=0.05, width=0.5, headwidth=7, headlength=7, 
                                                alpha=0.7),
                                fontsize=12, fontweight='bold', color=colormap(idx))              


            x_ticks = ax.get_xticks()  # Get the current tick positions
            ax.set_xticklabels([f'{x:.1f}' for x in x_ticks], rotation=45, ha='right') 
    #         # Get current x-axis tick locations
    #         x_ticks = ax.get_xticks()
    #         # Set new x-tick labels as integers
    #         ax.set_xticklabels([int(x) for x in x_ticks])

        # Set the global x and y limits
        if include_global_limit == False:
            pass
        else:
            ax.set_xlim(global_min_water_level-1, global_max_water_level+1)
            ax.set_ylim(global_min_exports-50000, global_max_exports+50000)

        # Customize the subplot
        if use_std == False:
            ax.set_xlabel('Water Level (mean)', fontsize=16)
            ax.set_ylabel('Cumulative Exports (ac-ft) (mean)', fontsize=16)
        else:
            ax.set_xlabel('Water Level (mean ± std)', fontsize=16)
            ax.set_ylabel('Cumulative Exports (ac-ft) (mean ± std)', fontsize=16)
        ax.set_title(f"PT Policy: {tmp_pt_policy}")

    #     # Add Water Level specific annotations (dashed vertical lines)
    #     levels = [6391, 6388]#, 6386]#, 6384]
    #     for level in levels:
    #         # Add vertical line
    #         ax.axvline(level, color='grey', linestyle='--')

        ax.grid(axis='x')

    #         if include_global_limit == False:
    #             ax.text(level + 0.1, ax.get_ylim()[0] + 20000, f'{level}', rotation=90, 
    #                     va='bottom', ha='left', color='black', fontsize=12)
    #         else:
    #             # Only add text if the level is within the global x-limits
    #             if global_min_water_level <= level <= global_max_water_level:
    #                 ax.text(level + 0.1, ax.get_ylim()[0] + 20000, f'{level}', rotation=90, 
    #                         va='bottom', ha='left', color='black', fontsize=12)

    # Adjust layout to prevent overlap
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    # Add a custom legend for the policies outside the plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.96, 0.90), fontsize=14)

    # Add an overall title for the figure
    fig.suptitle(f'Policy Performance Evaluation for Year: {year_of_interest}', fontsize=24, y=1.04)
    
def plot_gcm_percent_above_level_by_policy(Wrapped_or_Projections, policies_to_plot, pt_policies_to_plot, Variable, Water_Level_of_Interest, 
                               SSP_of_Interest, Policy_Outputs,key_word, Start_Year, End_Year):
    
    #key_word, Start_Year, End_Year = set_attributes(Wrapped_or_Projections)
    x_years = list(Policy_Outputs['year'].values)
    
    if Wrapped_or_Projections == 'Wrapped':
        tmp_x_yrs = x_years
        x_years = [x + Start_Year for x in tmp_x_yrs] 

    # Sequences or GCMs that will loop through
    clims = list(Policy_Outputs[key_word].values)
    policies = list(Policy_Outputs.policy.values)
    pt_policies = list(Policy_Outputs.PT_policy.values)

    if Wrapped_or_Projections == 'Projections':
        tmp_array = Policy_Outputs.sel(ssp=SSP_of_Interest)
    elif Wrapped_or_Projections == 'Wrapped':
        tmp_array = Policy_Outputs.copy(deep=True)

    # Determine number of PT policies to decide on subplot grid
    num_pt_policies = len(pt_policies_to_plot)

    # Calculate years since Start_Year
    years_since_start = abs(Start_Year - np.arange(Start_Year,End_Year+2))

    # Create subplots (one column per PT policy)
    fig, axes = plt.subplots(1, num_pt_policies, figsize=(8 * num_pt_policies, 6), sharey=True)

    # Ensure axes is always iterable (in case there is only one PT policy)
    if num_pt_policies == 1:
        axes = [axes]

    #colormap = ['red','green','blue','pink']
    colormap = cm.get_cmap('copper_r', len(policies_to_plot))
    for col_idx, tmp_pt_policy in enumerate(pt_policies_to_plot):
        ax = axes[col_idx]  # Select the correct subplot

        for policy_idx, tmp_policy in enumerate(policies_to_plot):

            color = colormap(policy_idx)  # Assign color based on policy index

            # Select the corresponding data
            tmp_tmp_array = tmp_array.sel(policy=tmp_policy, PT_policy=tmp_pt_policy)[Variable]

            # Calculate the percent of sequences or GCMs above the water level of interest
            percent_clims_above_water_level = (tmp_tmp_array >= Water_Level_of_Interest).sum(dim=key_word) / len(tmp_tmp_array[key_word]) * 100

            if 'U' in tmp_policy:
                linestyle = '--'
            else:
                linestyle = '-'

            # Plot the data, using the same color but different linestyles for each PT policy
            ax.plot(
                x_years,
                percent_clims_above_water_level,
                label=f'{tmp_policy}',#, PT: {tmp_pt_policy}' if len(pt_policies_to_plot) > 1 else f'{tmp_policy}',
                color=color,
                linestyle=linestyle  # Cycle through the linestyles
            )

        # Customize the subplot
        ax.set_ylabel('Percent (%)' if col_idx == 0 else '',fontsize=18)  # Only set ylabel on first subplot for shared y-axis
        ax.set_ylim(0, 100)
        ax.grid()

        # Select a subset of x_years for the x-ticks (e.g., every 5th year)
        subset_years = x_years[::10]  # Change 5 to the appropriate step to reduce overcrowding
        
        if End_Year in subset_years:
            pass
        else:
            end_yr_plus_1 = End_Year + 1
            subset_years.append(end_yr_plus_1)

        # Set the x-ticks to match the subset of years
        ax.set_xticks(subset_years)
        ax.set_xticklabels(subset_years, rotation=45)

        # Find the closest year for each tick location (after limiting the ticks to the subset)
        current_ticks = ax.get_xticks()
        closest_years = [x_years[np.abs(x_years - tick).argmin()] for tick in current_ticks]
        years_since = [int(year - Start_Year) for year in closest_years]

        # Add the second line for "Years since Start_Year" under the primary x-axis labels, excluding zero
        for tick, year_since in zip(current_ticks, years_since):
            #if year_since != 0:  # Skip the first year since (zero)
            ax.text(tick, -0.2, f"{year_since}", transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=14, color='gray')

        # Set title for each subplot
        ax.set_title(f"Post-Transition Policy: {tmp_pt_policy}")

        # Add a legend to each subplot
        if col_idx == 0:
            ax.legend(loc='upper left', title_fontsize=10)

    # Set a common title for the figure
    if Wrapped_or_Projections == 'Projections':
        fig.suptitle(f"Percent GCMs Above {Water_Level_of_Interest} ft, {SSP_of_Interest}", fontsize=22)
    else:
        fig.suptitle(f"Percent of Sequences Above {Water_Level_of_Interest} ft", fontsize=22)

    # Adjust layout
    plt.tight_layout()
    
def plot_gcm_percent_above_level_by_ssp(Wrapped_or_Projections, policies_to_plot, ssps_to_plot, Variable, pt_policy, Water_Level_of_Interest, Policy_Outputs,key_word, Start_Year, End_Year):

    #key_word, Start_Year, End_Year = set_attributes(Wrapped_or_Projections)
    x_years = list(Policy_Outputs['year'].values)

    # Sequences or GCMs that will loop through
    clims = list(Policy_Outputs[key_word].values)
    policies = list(Policy_Outputs.policy.values)

    #if Wrapped_or_Projections == 'Projections':
    #    tmp_array = Policy_Outputs.sel(ssp=SSP_of_Interest)
    #elif Wrapped_or_Projections == 'Wrapped':
    #    tmp_array = Policy_Outputs.copy(deep=True)

    # Calculate years since Start_Year
    years_since_start = abs(Start_Year - np.arange(Start_Year,End_Year+2))

    # Calculate number of SSPs to determine the subplot grid
    num_ssps = len(ssps_to_plot)

    # Create subplots (one column per SSP)
    fig, axes = plt.subplots(1, num_ssps, figsize=(8 * num_ssps, 6), sharey=True)

    # Ensure axes are always iterable (in case there's only one SSP)
    if num_ssps == 1:
        axes = [axes]

    #colormap = ['red','green','blue','pink']
    colormap = cm.get_cmap('copper_r', len(policies_to_plot))
    for col_idx, tmp_ssp in enumerate(ssps_to_plot):
        ax = axes[col_idx]  # Select the correct subplot

        # Update the data array to select the SSP of interest
        tmp_array_ssp = Policy_Outputs.sel(ssp=tmp_ssp,PT_policy=pt_policy)

        for policy_idx, tmp_policy in enumerate(policies_to_plot):
            color = colormap(policy_idx)  # Assign color based on SSP index

            # Select the corresponding data for the policy
            tmp_tmp_array = tmp_array_ssp.sel(policy=tmp_policy)[Variable]

            # Calculate the percent of sequences or GCMs above the water level of interest
            percent_clims_above_water_level = (tmp_tmp_array >= Water_Level_of_Interest).sum(dim=key_word) / len(tmp_tmp_array[key_word]) * 100

            if 'U' in tmp_policy:
                linestyle = '--'
            else:
                linestyle = '-'

            # Plot the data using the same color for the SSP but different linestyles for policies
            ax.plot(
                x_years,
                percent_clims_above_water_level,
                label=f'{tmp_policy}',
                color=color,
                linestyle=linestyle  # Cycle through the linestyles
            )

        # Customize the subplot
        ax.set_ylabel('Percent (%)' if col_idx == 0 else '', fontsize=18)  # Only set ylabel on first subplot for shared y-axis
        ax.set_ylim(0, 100)
        ax.grid()

        # Select a subset of x_years for the x-ticks (e.g., every 5th year)
        subset_years = x_years[::10]

        if End_Year not in subset_years:
            subset_years.append(End_Year)

        ax.set_xticks(subset_years)
        ax.set_xticklabels(subset_years, rotation=45)

        # Add a second line for "Years since Start_Year" under the primary x-axis labels
        current_ticks = ax.get_xticks()
        closest_years = [x_years[np.abs(x_years - tick).argmin()] for tick in current_ticks]
        years_since = [int(year - Start_Year) for year in closest_years]

        for tick, year_since in zip(current_ticks, years_since):
            ax.text(tick, -0.2, f"{year_since}", transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=14, color='gray')

        # Set title for each subplot
        ax.set_title(f"{tmp_ssp}",fontsize=18)

        # Add a legend to each subplot
        if col_idx == 0:
            ax.legend(loc='upper left', title_fontsize=10)

    # Set a common title for the figure
    fig.suptitle(f"Percent GCMs Above {Water_Level_of_Interest} ft for Post-Transition: {pt_policy}", fontsize=22)

    # Adjust layout
    plt.tight_layout()
    
### Get Runoff Year Type (RYT), SEF, and any other info of interest that is used to run WBM
def get_runoff_info(Wrapped_or_Projections, ERA5_Data, GCM_Data_Path, GCM_Member_List, SSPs_of_Interest,key_word, Start_Year, End_Year):
    
    #key_word, Start_Year, End_Year = set_attributes(Wrapped_or_Projections)
    
    if Wrapped_or_Projections == 'Wrapped':

        SEF_or_SRF_to_Predict_Grant_Outflow = 'SEF'
        SEF_or_SRF_for_Post_Transition_Flow_Requirement = 'SEF'
        hist_era5_df = pd.read_csv(f'{ERA5_Data}/ERA5_{SEF_or_SRF_to_Predict_Grant_Outflow}_Predicted_Flow_into_Mono_Lake.csv',index_col=0)
        
        wrapped_years = (Start_Year,End_Year)

        ### focusing on 1971-2019
        hist_era5_df = hist_era5_df[(hist_era5_df.index>=wrapped_years[0]) & (hist_era5_df.index<=wrapped_years[1])]

        for wrapped_start_yr in list(np.arange(wrapped_years[0],wrapped_years[1]+1)):

            first_years_of_wrapped_run = hist_era5_df[hist_era5_df.index>=wrapped_start_yr]
            last_years_of_wrapped_run = hist_era5_df[hist_era5_df.index<wrapped_start_yr]

            ### concatenate wrapped run data
            tmp_df = pd.concat([first_years_of_wrapped_run,last_years_of_wrapped_run])

            ### if choose to use integers from 0 to 50 (or whatever range from end - start year is)
            #tmp_df.index = np.arange(0,len(tmp_df))

            ### start of each sequence will vary, but index as start_year to end_year
            ### for consistency with checking policy export limit (for policies that are dynamic and keep track of years since start)
            tmp_df.index = np.arange(Start_Year,End_Year+1)

            Year_Type = tmp_df['RYT']
            Amended_License_Flow_Requirement = tmp_df[f'{SEF_or_SRF_for_Post_Transition_Flow_Requirement} Rush (ac-ft/year)']
            

            sys.exit('Need to update for Wrapped Runs')

    elif Wrapped_or_Projections == 'Projections':

        # Lists to store data for xarray dataset
        year_list = []
        year_type_list = []
        flow_req_list = []
        ssp_list = []
        gcm_list = []

        Historical_or_Dynamic_RYT = 'Dynamic'    ### will typically keep dynamic (updates RYT definition based on GCM)
        SEF_or_SRF_to_Predict_Grant_Outflow = 'SEF'
        SEF_or_SRF_for_Post_Transition_Flow_Requirement = 'SEF'

        for ssp in SSPs_of_Interest:

            for gcm in GCM_Member_List:

                tmp_df = pd.read_csv(f'{GCM_Data_Path}/{Historical_or_Dynamic_RYT}_RYT_{SEF_or_SRF_to_Predict_Grant_Outflow}/{gcm}_{ssp}.csv', index_col=0)

                ### slice years of interest (will get up to 2099 but do not have data for 2100 - note the data for 2099 is used to predict April 1st, 2100 water level though)
                tmp_df = tmp_df[(tmp_df.index>=Start_Year) & (tmp_df.index<=End_Year+1)]

                years = tmp_df.index

                Year_Type = tmp_df['RYT']
                Amended_License_Flow_Requirement = tmp_df[f'{SEF_or_SRF_for_Post_Transition_Flow_Requirement} Rush (ac-ft/year)']

                ### update gcm name to remove member and _hist
                gcm = gcm.split('_')[0]

                # Append data to the lists
                year_list.extend(years)
                year_type_list.extend(Year_Type.values)
                flow_req_list.extend(Amended_License_Flow_Requirement.values)
                ssp_list.extend([ssp] * len(years))
                gcm_list.extend([gcm] * len(years))

        # Create a pandas DataFrame from the lists
        data = pd.DataFrame({
            'year': year_list,
            'ssp': ssp_list,
            'GCM': gcm_list,
            'Year_Type': year_type_list,
            'Amended_License_Flow_Requirement': flow_req_list
        })

        # Convert the DataFrame to an xarray dataset
        ryt_sef_xarray = data.set_index(['ssp', 'GCM', 'year']).to_xarray()

    return ryt_sef_xarray

def percent_diff_gcm_above_level_by_ssp_policy(Wrapped_or_Projections, policies_to_plot, ssps_to_plot, Variable, pt_policy, Exports_Variable, 
                                               Water_Level_of_Interest, Policy_Outputs, ryt_sef_xarray,key_word, Start_Year, End_Year):
    

    # Define the number of years in a decade (10 years)
    years_per_decade = 10

    # Calculate years since Start_Year
    years_since_start = abs(Start_Year - np.arange(Start_Year, End_Year + 2))

    # Calculate number of decades
    num_decades = len(years_since_start) // years_per_decade

    # Create a colormap with enough colors for all decades
    colormap = cm.get_cmap('copper_r', num_decades)

    # Update the grid size based on the number of policies
    num_policies = len(policies_to_plot)
    grid_size = int(np.ceil(np.sqrt(num_policies)))  # Square grid layout

    # Create subplots for each policy in a grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 18), sharey=True)

    # Flatten axes to easily iterate
    axes = axes.flatten()

    # Bar width and spacing adjustments
    bar_width = 0.15  # Smaller width for more spacing between bars
    gap_width = 0.2   

    # Variables to track global min and max for export differences
    global_export_min = float('inf')
    global_export_max = float('-inf')

    # Special case for A1: Calculate export range across all decades for A1
    a1_export_min = float('inf')
    a1_export_max = float('-inf')

    for ssp_idx, tmp_ssp in enumerate(ssps_to_plot):
        a1_export_array = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy='A1')[Exports_Variable]

        for decade in range(num_decades):
            start_idx = decade * years_per_decade
            end_idx = start_idx + years_per_decade

            # Calculate the slice of exports for the current decade
            a1_export_decade = a1_export_array.isel(year=slice(start_idx, end_idx))

            # Get the year types for the current SSP and decade
            tmp_ryt_array = ryt_sef_xarray.sel(ssp=tmp_ssp)
            tmp_ryt_array_decade = tmp_ryt_array.isel(year=slice(start_idx, end_idx))['Year_Type']

            # Apply masks for dry and wet year types
            dry_mask = np.isin(tmp_ryt_array_decade, dry_types)
            wet_mask = np.isin(tmp_ryt_array_decade, wet_types)

            # Compute average exports for dry and wet year types
            dry_exports = a1_export_decade.where(dry_mask)
            wet_exports = a1_export_decade.where(wet_mask)

            # Calculate dry and wet year min and max for A1
            dry_exports_mean = dry_exports.mean(dim=['GCM', 'year'], skipna=True).values
            wet_exports_mean = wet_exports.mean(dim=['GCM', 'year'], skipna=True).values

            # Update export min and max for A1
            a1_export_min = min(a1_export_min, dry_exports_mean, wet_exports_mean)
            a1_export_max = max(a1_export_max, dry_exports_mean, wet_exports_mean)

    # First pass to calculate global export min and max for other policies (excluding A1 for difference calculations)
    for policy_idx, tmp_policy in enumerate(policies_to_plot[1:]):  # Skip A1 for difference calculations
        for ssp_idx, tmp_ssp in enumerate(ssps_to_plot):
            export_policy = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy=tmp_policy)[Exports_Variable]
            export_baseline = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy='A1')[Exports_Variable]

            for decade in range(num_decades):
                start_idx = decade * years_per_decade
                end_idx = start_idx + years_per_decade

                # Extract the decade data for the policy and baseline (A1)
                export_policy_decade = export_policy.isel(year=slice(start_idx, end_idx))
                export_baseline_decade = export_baseline.isel(year=slice(start_idx, end_idx))

                # Get the year types for the current SSP and decade
                tmp_ryt_array = ryt_sef_xarray.sel(ssp=tmp_ssp)
                tmp_ryt_array_decade = tmp_ryt_array.isel(year=slice(start_idx, end_idx))['Year_Type']

                # Apply masks for dry and wet year types
                dry_mask = np.isin(tmp_ryt_array_decade, dry_types)
                wet_mask = np.isin(tmp_ryt_array_decade, wet_types)

                # Compute export differences for dry and wet year types
                dry_export_policy = export_policy_decade.where(dry_mask)
                dry_export_baseline = export_baseline_decade.where(dry_mask)
                dry_export_diff = (dry_export_policy.mean(dim=['GCM', 'year'], skipna=True) - dry_export_baseline.mean(dim=['GCM', 'year'], skipna=True)).values

                wet_export_policy = export_policy_decade.where(wet_mask)
                wet_export_baseline = export_baseline_decade.where(wet_mask)
                wet_export_diff = (wet_export_policy.mean(dim=['GCM', 'year'], skipna=True) - wet_export_baseline.mean(dim=['GCM', 'year'], skipna=True)).values

                # Update global min and max for export differences based on dry and wet years
                global_export_min = min(global_export_min, dry_export_diff, wet_export_diff)
                global_export_max = max(global_export_max, dry_export_diff, wet_export_diff)


    # Round a1_export_max to the nearest 1000 and multiply by -1 for global_export_min
    a1_export_max = np.round(a1_export_max, -3)
    global_export_min = -1 * a1_export_max

    # Second pass to plot the data
    for policy_idx, tmp_policy in enumerate(policies_to_plot):
        ax = axes[policy_idx]  # Select the correct subplot for each policy

        # Create secondary y-axis for exports
        ax2 = ax.twinx()

        # Positions for each SSP on the x-axis, with increased gap
        positions = np.arange(len(ssps_to_plot)) * (bar_width * num_decades + gap_width)


        if tmp_policy == 'A1':
            # Iterate over the SSPs and plot actual percent of GCMs and exports for A1
            for ssp_idx, tmp_ssp in enumerate(ssps_to_plot):
                a1_array_ssp = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy='A1')[Variable]
                a1_export_array = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy='A1')[Exports_Variable]

                # Decades' percent of GCMs at or above the water level and average exports for the current SSP
                a1_percent_above = []
                dry_exports_mean_list = []
                wet_exports_mean_list = []

                for decade in range(num_decades):
                    # Slice data for the current decade
                    start_idx = decade * years_per_decade
                    end_idx = start_idx + years_per_decade
                    a1_decade = a1_array_ssp.isel(year=slice(start_idx, end_idx))
                    a1_export_decade = a1_export_array.isel(year=slice(start_idx, end_idx))

                    # Calculate the percent of GCMs at or above the water level of interest
                    percent_above_a1 = (a1_decade >= Water_Level_of_Interest).sum(dim='GCM') / len(a1_decade.GCM) * 100
                    a1_percent_above.append(percent_above_a1.mean(dim='year').values)

                    ################################# RYT ###############################
                    # (Dry, Dry-Normal, Normal) vs > Normal (Wet-Normal, Wet, Extreme-Wet)
                    ## get RYT for ssp of interest
                    tmp_ryt_array = ryt_sef_xarray.sel(ssp=tmp_ssp)
                    ## get slice of years of interest
                    tmp_ryt_array_decade = tmp_ryt_array.isel(year=slice(start_idx, end_idx))
                    ## intersted in year_type
                    tmp_ryt_array_decade = tmp_ryt_array_decade['Year_Type']

                    # Mask for "Dry", "Dry-Normal", and "Normal" year types
                    dry_mask = np.isin(tmp_ryt_array_decade, dry_types)

                    # Mask for "Wet-Normal", "Wet", and "Extreme-Wet" year types
                    wet_mask = np.isin(tmp_ryt_array_decade, wet_types)

                    # Compute average exports for "Dry", "Dry-Normal", and "Normal" year types
                    dry_exports = a1_export_decade.where(dry_mask)  # Apply the mask
                    dry_exports_mean = dry_exports.mean(dim=['GCM', 'year'], skipna=True)  # Compute the mean over GCM and year
                    dry_exports_mean_list.append(dry_exports_mean.values)

                    # Compute average exports for "Wet-Normal", "Wet", and "Extreme-Wet" year types
                    wet_exports = a1_export_decade.where(wet_mask)  # Apply the mask
                    wet_exports_mean = wet_exports.mean(dim=['GCM', 'year'], skipna=True)  # Compute the mean over GCM and year
                    wet_exports_mean_list.append(wet_exports_mean.values)

                # Plot bars for the percent of GCMs at or above the water level
                for decade in range(num_decades):
                    bar_pos = positions[ssp_idx] + decade * bar_width
                    bar_height = a1_percent_above[decade]

                    # Plot percent above water level bars
                    ax.bar(bar_pos, bar_height, width=bar_width, label=f'Decade {decade + 1}' if ssp_idx == 0 else "", color=colormap(decade))

                    # Add decade label above the bar
                    ax.text(bar_pos, bar_height + 0.25, f'{decade + 1}', ha='center', va='bottom', fontsize=11)

                # Plot dry_exports_mean and wet_exports_mean bars on the secondary y-axis
                for decade in range(num_decades):
                    bar_pos = positions[ssp_idx] + decade * bar_width
                    dry_export_mean = dry_exports_mean_list[decade]
                    wet_export_mean = wet_exports_mean_list[decade]

                    # Plot wet_exports_mean as blue
                    if np.isfinite(wet_export_mean):
                        ax2.bar(bar_pos, wet_export_mean, width=bar_width, alpha=0.3, color='blue')

                    # Plot dry_exports_mean as red with some alpha
                    if np.isfinite(dry_export_mean):
                        ax2.bar(bar_pos, dry_export_mean, width=bar_width, alpha=0.5, edgecolor='red', fill=False, linewidth=2.0)

    #                 # Plot wet_exports_mean as blue, stacked on top of dry_exports_mean
    #                 if np.isfinite(wet_export_mean) and np.isfinite(dry_export_mean):
    #                     ax2.bar(bar_pos, wet_export_mean, bottom=dry_export_mean, width=bar_width, alpha=0.5, color='blue')
    #                 elif np.isfinite(wet_export_mean):
    #                     ax2.bar(bar_pos, wet_export_mean, width=bar_width, alpha=0.5, color='blue')


            ax.text(-1.0,0,f'Percent GCMs Reaching {Water_Level_of_Interest}',rotation=90,fontsize=16,color='grey')
            ax.text(4.5,10,f'Exports (ac-ft)',rotation=270,fontsize=16,color='grey')
            # Set the secondary y-axis limits for exports based on the range for A1
            ax2.set_ylim(0, a1_export_max)

        else:
            # For other policies, calculate differences from A1 and percent difference
            for ssp_idx, tmp_ssp in enumerate(ssps_to_plot):
                # Select the data for the policy and A1
                export_policy = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy=tmp_policy)[Exports_Variable]
                export_baseline = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy='A1')[Exports_Variable]
                tmp_array_ssp = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy=tmp_policy)[Variable]
                baseline_array = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy='A1')[Variable]

                dry_exports_diff_list = []
                wet_exports_diff_list = []
                decade_means = []

                for decade in range(num_decades):
                    start_idx = decade * years_per_decade
                    end_idx = start_idx + years_per_decade
                    export_policy_decade = export_policy.isel(year=slice(start_idx, end_idx))
                    export_baseline_decade = export_baseline.isel(year=slice(start_idx, end_idx))

                    # Calculate percent difference in GCMs reaching the water level of interest
                    tmp_policy_decade = tmp_array_ssp.isel(year=slice(start_idx, end_idx))
                    baseline_decade = baseline_array.isel(year=slice(start_idx, end_idx))
                    percent_above_policy = (tmp_policy_decade >= Water_Level_of_Interest).sum(dim='GCM') / len(tmp_policy_decade.GCM) * 100
                    percent_above_baseline = (baseline_decade >= Water_Level_of_Interest).sum(dim='GCM') / len(baseline_decade.GCM) * 100
                    percent_diff = (percent_above_policy - percent_above_baseline).mean(dim='year').values
                    decade_means.append(percent_diff)

                    # Get the year types for the current SSP and decade
                    tmp_ryt_array = ryt_sef_xarray.sel(ssp=tmp_ssp)
                    tmp_ryt_array_decade = tmp_ryt_array.isel(year=slice(start_idx, end_idx))['Year_Type']

                    # Apply masks for dry and wet year types
                    dry_mask = np.isin(tmp_ryt_array_decade, dry_types)
                    wet_mask = np.isin(tmp_ryt_array_decade, wet_types)

                    # Compute the export differences for dry and wet year types relative to A1
                    dry_export_policy = export_policy_decade.where(dry_mask)
                    dry_export_baseline = export_baseline_decade.where(dry_mask)
                    dry_export_diff = (dry_export_policy.mean(dim=['GCM', 'year']) - dry_export_baseline.mean(dim=['GCM', 'year']))

                    wet_export_policy = export_policy_decade.where(wet_mask)
                    wet_export_baseline = export_baseline_decade.where(wet_mask)
                    wet_export_diff = (wet_export_policy.mean(dim=['GCM', 'year']) - wet_export_baseline.mean(dim=['GCM', 'year']))

                    dry_exports_diff_list.append(dry_export_diff.values)
                    wet_exports_diff_list.append(wet_export_diff.values)

                # Plot percent difference bars for GCMs reaching the water level (primary y-axis)
                for decade in range(num_decades):
                    bar_pos = positions[ssp_idx] + decade * bar_width
                    bar_height = decade_means[decade]

                    ax.bar(bar_pos, bar_height, width=bar_width, label=f'Decade {decade + 1}' if ssp_idx == 0 else "", color=colormap(decade))

                    # Add decade label above the bar
                    ax.text(bar_pos, bar_height + 0.25, f'{decade + 1}', ha='center', va='bottom', fontsize=11)

                # Plot dry_exports_diff and wet_exports_diff bars on the secondary y-axis
                for decade in range(num_decades):
                    bar_pos = positions[ssp_idx] + decade * bar_width
                    dry_export_diff = dry_exports_diff_list[decade]
                    wet_export_diff = wet_exports_diff_list[decade]

                    # Plot dry_exports_diff as red outline
                    if np.isfinite(dry_export_diff):
                        ax2.bar(bar_pos, dry_export_diff, width=bar_width, alpha=0.5, edgecolor='red', fill=False, linewidth=2.0)

                    # Plot wet_exports_diff as blue
                    if np.isfinite(wet_export_diff):
                        ax2.bar(bar_pos, wet_export_diff, width=bar_width, alpha=0.3, color='blue')

            # Customize secondary y-axis for exports
            ax2.set_ylim(global_export_min, global_export_max)  # Apply consistent y-limits across other subplots
            ax2.grid(False)

        # Add a baseline line for 0% change
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # Customize the subplot
        ax.set_xticks(positions + bar_width * (num_decades - 1) / 2)  # Center the tick labels
        ax.set_xticklabels(ssps_to_plot)
        ax.set_title(f'Policy {tmp_policy}', fontsize=16, fontweight='bold')
        if tmp_policy != 'A1':
            ax.grid(axis='y')

    # Hide unused subplots
    for i in range(len(policies_to_plot), grid_size * grid_size):
        fig.delaxes(axes[i])

    # Set common titles and labels
    fig.suptitle(f"Percent Difference in GCMs reaching Water Level {Water_Level_of_Interest} ft (relative to Existing Policy)\nPost-Transition: {pt_policy}", fontsize=22, y=1.0)
    fig.text(-0.02, 0.5, 'Percent Difference (%) from Existing Policy (A1)', va='center', rotation='vertical', fontsize=22)
    fig.text(1.005, 0.5, 'Difference in Exports (ac-ft) from Existing Policy (A1)', va='center', rotation=270, fontsize=22)

    # Adjust layout
    plt.tight_layout()
    
def scatter_plot_dry_wet_by_ssp_policy(Wrapped_or_Projections, policies_to_plot, ssps_to_plot, Variable, pt_policies, 
                                       Water_Level_of_Interest, start_year_index, end_year_index, Policy_Outputs, ryt_sef_xarray,key_word, Start_Year, End_Year):

    #key_word, Start_Year, End_Year = set_attributes(Wrapped_or_Projections)
    
    ### how the dry vs wet year_types will be split
    dry_types = ['Dry', 'Dry-Normal', 'Normal']
    wet_types = ['Wet-Normal', 'Wet', 'Extreme-Wet']

    # if start_year_index >= 15:
    #     policies_to_plot = ['A1','A2','A3','A4','A5','A6','A10','U4']

    color_map = plt.cm.get_cmap('RdYlGn')  # Colormap for marker colors


    # Set specific x- and y-ticks for export values (0, 4500, 9000, 13500, 16000)
    export_ticks = [0, 4500, 9000, 13500, 16000]

    tmp_start_yr_label = Start_Year + start_year_index
    tmp_end_yr_label = Start_Year + end_year_index

    # Create a grid of subplots
    nrows = len(ssps_to_plot)
    ncols = len(pt_policies)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8), sharex=True, sharey=True)

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Prepare variables to hold min and max percent values across all pt_policies for each SSP (for the colorbar and marker scaling)
    percent_min_max = {}

    # First pass: Calculate the global min and max percent for each SSP (for colorbar and marker scaling)
    for ssp in ssps_to_plot:
        percent_min_max[ssp] = [100, 0]  # Initialize min and max

        for pt_policy in pt_policies:
            for tmp_policy in policies_to_plot:
                # Select the data for the policy, SSP, and PT_policy
                tmp_array_ssp = Policy_Outputs.sel(ssp=ssp, PT_policy=pt_policy, policy=tmp_policy)[Variable]
                water_level_years = tmp_array_ssp.isel(year=slice(start_year_index, end_year_index))

                # Calculate the percentage of GCMs above the water level of interest
                percent_above = ((water_level_years >= Water_Level_of_Interest).sum(dim='GCM') / len(water_level_years.GCM) * 100).mean(dim='year').values

                # Update the min and max percent for the current SSP
                percent_min_max[ssp][0] = min(percent_min_max[ssp][0], percent_above)
                percent_min_max[ssp][1] = max(percent_min_max[ssp][1], percent_above)

    # Iterate over SSPs and PT_policies to create a grid of scatter plots
    for ssp_idx, ssp in enumerate(ssps_to_plot):
        # Normalize the percent values for the color and marker size based on the global range for this SSP
        percent_min, percent_max = percent_min_max[ssp]
        percent_min = np.floor(percent_min_max[ssp][0] / 5) * 5  # Round the minimum to the nearest 5 (down)
        percent_max = np.ceil(percent_min_max[ssp][1] / 5) * 5   # Round the maximum to the nearest 5 (up)
        norm_percent = mcolors.Normalize(vmin=percent_min, vmax=percent_max)  # Normalized per SSP

        for pt_policy_idx, pt_policy in enumerate(pt_policies):
            ax = axes[ssp_idx * ncols + pt_policy_idx]  # Select subplot

            # Set the x- and y-ticks for the scatter plot
            ax.set_xticks(export_ticks)
            ax.set_yticks(export_ticks)

            # Rotate the x-axis tick labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            # Prepare lists to store data for the scatter plot
            dry_exports_x = []
            wet_exports_y = []
            percent_above_water_level = []
            policy_labels = []  # For labeling the policies on the scatter plot

            # Iterate over each policy
            for tmp_policy in policies_to_plot:
                # Select the data for the policy, SSP, and PT_policy
                export_policy = Policy_Outputs.sel(ssp=ssp, PT_policy=pt_policy, policy=tmp_policy)['Exports']
                tmp_array_ssp = Policy_Outputs.sel(ssp=ssp, PT_policy=pt_policy, policy=tmp_policy)[Variable]

                # Extract year slice for user-defined range
                export_policy_years = export_policy.isel(year=slice(start_year_index, end_year_index))
                water_level_years = tmp_array_ssp.isel(year=slice(start_year_index, end_year_index))

                # Get the year types for the current SSP and year range
                tmp_ryt_array = ryt_sef_xarray.sel(ssp=ssp)
                tmp_ryt_array_years = tmp_ryt_array.isel(year=slice(start_year_index, end_year_index))['Year_Type']

                # Apply masks for dry and wet year types
                dry_mask = np.isin(tmp_ryt_array_years, dry_types)
                wet_mask = np.isin(tmp_ryt_array_years, wet_types)

                # Compute average exports for dry and wet year types
                dry_exports_mean = export_policy_years.where(dry_mask).mean(dim=['GCM', 'year'], skipna=True).values
                wet_exports_mean = export_policy_years.where(wet_mask).mean(dim=['GCM', 'year'], skipna=True).values

                # Calculate the percentage of GCMs above the water level of interest
                percent_above = ((water_level_years >= Water_Level_of_Interest).sum(dim='GCM') / len(water_level_years.GCM) * 100).mean(dim='year').values

                # Append the calculated values to the lists
                dry_exports_x.append(dry_exports_mean)
                wet_exports_y.append(wet_exports_mean)
                percent_above_water_level.append(percent_above)
                policy_labels.append(tmp_policy)  # Append the policy label

            # Convert lists to numpy arrays for scatter plotting
            dry_exports_x = np.array(dry_exports_x)
            wet_exports_y = np.array(wet_exports_y)
            percent_above_water_level = np.array(percent_above_water_level)

            # Marker sizes: Consistent scaling across PT_policies for each SSP, based on the global percent range
            marker_sizes = 1000 * (percent_above_water_level / percent_max)**2  # Larger marker size with a consistent scaling
            marker_colors = color_map(norm_percent(percent_above_water_level))  # Color by percentage

            # Create the scatter plot in the current subplot
            scatter = ax.scatter(dry_exports_x, wet_exports_y, s=500, c=percent_above_water_level, cmap=color_map, edgecolor='black', alpha=0.5)

            ax.set_xlim(-1000,17000)
            ax.set_ylim(-1000,17000)

            ax.grid(color='grey',alpha=0.5)

            # Add text labels directly on top of each marker
            for i, policy in enumerate(policy_labels):
                ax.text(dry_exports_x[i], wet_exports_y[i], policy, 
                        ha='center', va='center', fontsize=10, color='black')

            # Set titles for each subplot, but only in the first row
            if ssp_idx == 0:  # Only set titles for the first row
                ax.set_title(f'Post-Transition: {pt_policy}', fontsize=16,pad=10)

        # Add a unique colorbar for each SSP in the last column (right of the plot) based on the global range
        colorbar = fig.colorbar(mappable=plt.cm.ScalarMappable(norm=norm_percent, cmap=color_map), 
                                ax=axes[ssp_idx * ncols + ncols - 1], 
                                label='% GCMs Above Water Level', 
                                orientation='vertical', fraction=0.05, pad=0.04)

        # Define the tick values for the colorbar (every 5% between percent_min and percent_max)
        tick_values = np.linspace(percent_min, percent_max, num=5)  # Adjust 'num' to change the number of ticks
        colorbar.set_ticks(tick_values)

        # Optionally, format the tick labels (you can change this as needed)
        colorbar.set_ticklabels([f'{int(tick)}%' for tick in tick_values])

        # Set the label for the colorbar
        colorbar.set_label(f'% GCMs', fontsize=14)


    # Set common x and y labels
    fig.text(0.5, -0.03, 'Average Dry Year Exports (ac-ft)', ha='center', fontsize=16)
    fig.text(-0.09, 0.5, 'Average Wet Year Exports (ac-ft)', va='center', rotation='vertical', fontsize=16)

    fig.text(-0.06, 0.7, 'ssp245', va='center', fontsize=16)
    fig.text(-0.06, 0.45, 'ssp370', va='center', fontsize=16)
    fig.text(-0.06, 0.2, 'ssp585', va='center', fontsize=16)

    # # Dynamically add the SSP labels based on the SSPs being plotted
    # for idx, ssp in enumerate(ssps_to_plot):
    #     # Calculate the height factor to place the labels in the middle of each row
    #     pos_y = 1 - (idx + 0.575) / nrows  # Center labels in each row
    #     fig.text(-0.06, pos_y, ssp, va='center', fontsize=16, transform=fig.transFigure)

    # Set the main title of the figure
    fig.suptitle(f'Average Exports (Dry vs Wet Years) and Percent of GCMs at or above {Water_Level_of_Interest} ft\n{tmp_start_yr_label}-{tmp_end_yr_label} (Years {start_year_index}-{end_year_index} since {Start_Year})', fontsize=16,y=0.95)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

### Table of Important Metrics
def generate_table_of_metrics(Wrapped_or_Projections, SSP_of_Interest, pt_policies_to_plot, Policy_Outputs):
    
    key_word, Start_Year, End_Year = set_attributes(Wrapped_or_Projections)
    
    ### Define Four Years to Check
    if Wrapped_or_Projections == 'Wrapped':
        year_check = [10,20,30,40]
    elif Wrapped_or_Projections == 'Projections':
        year_check = [2030,2050,2070,2100]
    ### User-Defined ###

    Policy_list = []
    for elem in list(Policy_Outputs.policy.values):
        for elem_2 in pt_policies_to_plot:
            Policy_list.append(f'{elem}, PT: {elem_2}')

    metric_list = [f"{year_check[0]}_WL_Mean (ft)",f"{year_check[0]}_WL_STD",f"{year_check[0]}_WL_Min",f"{year_check[0]}_WL_Max",
                   f"{year_check[1]}_WL_Mean (ft)",f"{year_check[1]}_WL_STD",f"{year_check[1]}_WL_Min",f"{year_check[1]}_WL_Max",
                   f"{year_check[2]}_WL_Mean (ft)",f"{year_check[2]}_WL_STD",f"{year_check[2]}_WL_Min",f"{year_check[2]}_WL_Max",
                   f"{year_check[3]}_WL_Mean (ft)",f"{year_check[3]}_WL_STD",f"{year_check[3]}_WL_Min",f"{year_check[3]}_WL_Max",
                    "Transition_Time (Yrs)","Transition_STD", "%_Transition", 
                    "Avg_Exports (ac-ft)","Cum_Exports (1000s ac-ft)"]

    Policy_Summary_Table = pd.DataFrame(index = Policy_list, columns = metric_list)

    for year_tmp in year_check:

        for policy_tmp in list(Policy_Outputs.policy.values):

            for pt_policy_tmp in pt_policies_to_plot:

                if Wrapped_or_Projections == 'Wrapped':
                    ### no ssp
                    level_mean = np.round(Policy_Outputs["Water_Level"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp,year=year_tmp).mean(key_word).values,1)
                    level_std = np.round(Policy_Outputs["Water_Level"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp,year=year_tmp).std(key_word).values,1)
                    level_min = np.round(Policy_Outputs["Water_Level"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp,year=year_tmp).min(key_word).values,1)
                    level_max = np.round(Policy_Outputs["Water_Level"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp,year=year_tmp).max(key_word).values,1)
                else:
                    level_mean = np.round(Policy_Outputs["Water_Level"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp,ssp=SSP_of_Interest,year=year_tmp).mean(key_word).values,1)
                    level_std = np.round(Policy_Outputs["Water_Level"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp,ssp=SSP_of_Interest,year=year_tmp).std(key_word).values,1)
                    level_min = np.round(Policy_Outputs["Water_Level"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp,ssp=SSP_of_Interest,year=year_tmp).min(key_word).values,1)
                    level_max = np.round(Policy_Outputs["Water_Level"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp,ssp=SSP_of_Interest,year=year_tmp).max(key_word).values,1)

                Policy_Summary_Table[f"{year_tmp}_WL_Mean (ft)"][f'{policy_tmp}, PT: {pt_policy_tmp}']=level_mean
                Policy_Summary_Table[f"{year_tmp}_WL_STD"][f'{policy_tmp}, PT: {pt_policy_tmp}']=level_std
                Policy_Summary_Table[f"{year_tmp}_WL_Min"][f'{policy_tmp}, PT: {pt_policy_tmp}']=level_min
                Policy_Summary_Table[f"{year_tmp}_WL_Max"][f'{policy_tmp}, PT: {pt_policy_tmp}']=level_max


    for policy_tmp in list(Policy_Outputs.policy.values):

        for pt_policy_tmp in pt_policies_to_plot:

            if Wrapped_or_Projections == 'Wrapped':
                trans_mean = np.round(Policy_Outputs["Transition_Time"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp).mean(key_word).values,1)
                trans_std = np.round(Policy_Outputs["Transition_Time"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp).std(key_word).values,1)
                trans_check = Policy_Outputs["Transition_Time"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp).values
            elif Wrapped_or_Projections == 'Projections':
                trans_mean = np.round(Policy_Outputs["Transition_Time"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp,ssp=SSP_of_Interest).mean(key_word).values,1)
                trans_std = np.round(Policy_Outputs["Transition_Time"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp,ssp=SSP_of_Interest).std(key_word).values,1)
                trans_check = Policy_Outputs["Transition_Time"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp,ssp=SSP_of_Interest).values

            ### HARD-CODED YEARS
            tmp_num_years = (End_Year - Start_Year)+1
            trans_check = (trans_check < tmp_num_years).sum()
            trans_perc = (trans_check/11) * 100

            trans_perc_np = np.array(trans_perc)
            trans_perc_np = np.round(trans_perc_np,1)

            Policy_Summary_Table[f"Transition_Time (Yrs)"][f'{policy_tmp}, PT: {pt_policy_tmp}']=trans_mean
            Policy_Summary_Table[f"Transition_STD"][f'{policy_tmp}, PT: {pt_policy_tmp}']=trans_std
            Policy_Summary_Table[f"%_Transition"][f'{policy_tmp}, PT: {pt_policy_tmp}']=trans_perc_np

            ### Also get Exports
            if Wrapped_or_Projections == 'Wrapped':
                exp_mean = Policy_Outputs["Exports"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp).mean("year").mean(key_word).values
                exp_sum = Policy_Outputs["Exports"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp).sum("year").mean(key_word).values
            elif Wrapped_or_Projections == 'Projections':
                exp_mean = Policy_Outputs["Exports"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp,ssp=SSP_of_Interest).mean("year").mean(key_word).values
                exp_sum = Policy_Outputs["Exports"].sel(policy=policy_tmp,PT_policy=pt_policy_tmp,ssp=SSP_of_Interest).sum("year").mean(key_word).values

            exp_mean_np = np.array(exp_mean)
            exp_mean_np = np.round(exp_mean_np,1)
            exp_sum_np = np.array(exp_sum)
            exp_sum_np = exp_sum_np/1000
            exp_sum_np = np.round(exp_sum_np,1)

            Policy_Summary_Table[f"Avg_Exports (ac-ft)"][f'{policy_tmp}, PT: {pt_policy_tmp}']=exp_mean_np
            Policy_Summary_Table[f"Cum_Exports (1000s ac-ft)"][f'{policy_tmp}, PT: {pt_policy_tmp}']=exp_sum_np
            Policy_Summary_Table.to_csv('Policy_Summary_Table.csv', index=False)
            
            return Policy_Summary_Table

# Function to create a checkbox-based selector with "Select All" and "Deselect All"
def create_policy_selector(policies, title_text, default_select_all=False):
    """
    Create a checkbox-based selector for a list of policies with "Select All" and "Deselect All" options.
    
    Args:
        policies (list): List of policy names to display.
        title_text (str): Title to display above the checkboxes.
        default_select_all (bool): Whether all checkboxes should be selected by default.
        
    Returns:
        tuple: A tuple containing the VBox and a function to retrieve the selected policies.
    """
    # Create checkboxes for each policy with default selection state
    policy_checkboxes = [
        widgets.Checkbox(
            value=default_select_all,  # Use the default_select_all flag
            description=policy,
            layout=widgets.Layout(margin='0 0 0 5px', width='auto')
        ) for policy in policies
    ]
    
    # Dynamic list to store selected policies
    selected_policies = []

    # Function to retrieve selected policies
    def get_selected_policies():
        return [cb.description for cb in policy_checkboxes if cb.value]
    
    # Create an interactive output with an initial default message
    output = widgets.Output()
    with output:
        selected_policies.clear()
        selected_policies.extend(get_selected_policies())  # Initialize selected_policies
        print(f"Selected {title_text}: {selected_policies}")
    
    # Function to update output interactively
    def on_checkbox_change(_):
        with output:
            output.clear_output()
            selected_policies.clear()
            selected_policies.extend(get_selected_policies())
            print(f"Selected {title_text}: {selected_policies}")
    
    # Attach observers to all checkboxes
    for cb in policy_checkboxes:
        cb.observe(on_checkbox_change, names='value')
    
    # Function to select all policies
    def select_all(_):
        for cb in policy_checkboxes:
            cb.value = True  # Set all checkboxes to checked
    
    # Function to deselect all policies
    def deselect_all(_):
        for cb in policy_checkboxes:
            cb.value = False  # Set all checkboxes to unchecked
    
    # Buttons for "Select All" and "Deselect All"
    select_all_button = widgets.Button(description="Select All")
    deselect_all_button = widgets.Button(description="Deselect All")
    select_all_button.on_click(select_all)
    deselect_all_button.on_click(deselect_all)
    
    # Group the checkboxes in a vertical layout with a narrower box
    checkbox_group = widgets.VBox(
        policy_checkboxes,
        layout=widgets.Layout(
            border='1px solid black',
            padding='5px',
            width='275px',
            align_items='flex-start'
        )
    )
    
    # Add a larger title with increased font size and enforced left alignment
    title = widgets.HTML(f"<b style='font-size:16px;'>{title_text}</b>")
    
    # Combine title, buttons, checkbox group, and output into a single VBox
    selector_widget = widgets.VBox([title, widgets.HBox([select_all_button, deselect_all_button]), checkbox_group, output])
    
    return selector_widget, selected_policies




#####################

### Create Policy

# Main function to create a new policy
def create_user_policy(Policy_Data):
    # Set the number of phases for the user-defined policy
    num_phase = widgets.IntText(
        value=1,
        description='Number of phases:',
        style={'description_width': 'initial'},
        disabled=False
    )

    # Set the name for the user-defined policy
    pol_name = widgets.Text(
        value='',
        description='Name of policy:',
        style={'description_width': 'initial'},
        disabled=False
    )

    # Buttons for actions
    define_button = widgets.Button(description="1) Define Policy")  # Updated button label
    update_button = widgets.Button(description="2) Review Policy")
    add_button = widgets.Button(description="3) Add Policy")  # New "Add" button
    define_button.layout.display = '230px'
    update_button.layout.display = '230px'  # Initially hidden
    add_button.layout.display = '230px'  # Initially hidden

    # Output widget to display results
    output_display = widgets.Output()

    # Track current policy configuration
    current_policy_data = {"number_of_phase": None, "policy_name": None, "out3": None, "data_array": None}


    def process_policy(number_of_phase, out3):
        """
        Process user-defined policy based on the widget structure from config_user_policy.
        Ensures water levels are in strictly increasing order (no duplicates).
        """
        user_dict = {}
        dyn_type = None
        phase_threshs = [] if number_of_phase > 1 else None

        try:
            # Iterate through each phase to extract water level thresholds and export amounts
            for i in range(number_of_phase):
                lev_dict = {}
                phase_widget = out3.children[i]  # Get the widget for the current phase

                # Extract water level thresholds
                thresh_widget = phase_widget.children[0].children[0]  # Water Level Thresholds text widget
                thresh_str = thresh_widget.value
                threshs = [int(item) for item in thresh_str.split()]  # Convert thresholds to integers

                # Validate that thresholds are strictly increasing (no duplicates)
                if threshs != sorted(set(threshs)):
                    display(f"Error: Phase {i + 1} water level thresholds are not strictly increasing. Please redefine.")
                    return None

                # Extract export amounts for each threshold
                export_tab = phase_widget.children[1].children[0]  # Tab widget for export amounts
                for j, thresh in enumerate(threshs):
                    ryt_values = []
                    for k in range(6):  # Assuming 6 year types
                        year_type_widget = export_tab.children[k]  # Text widget for each year type
                        value_str = year_type_widget.value
                        values = [int(item) for item in value_str.split()]
                        ryt_values.append(values[j])  # Collect export amounts for this threshold
                    lev_dict[thresh] = ryt_values

                user_dict[i + 1] = lev_dict

            # Extract dynamic type and phase thresholds if multiple phases
            if number_of_phase > 1:
                dyn_type = out3.children[-2].value  # Dynamic type dropdown widget
                phase_thresh_widget = out3.children[-1]  # Accordion for phase thresholds
                phase_threshs = [phase_thresh_widget.children[j].value for j in range(len(phase_thresh_widget.children))]

        except Exception as e:
            display(f"Error processing policy: {e}")
            return None

        # Create data array using the parsed user input
        return create_dataarray(user_dict, dyn_type, phase_threshs)

    def on_define(_):
        with output_display:
            output_display.clear_output()
            #print("Define button clicked")

            # Retrieve the current values from the widgets
            policy_name = pol_name.value
            number_of_phases = num_phase.value

            # Validate the number of phases
            if not isinstance(number_of_phases, int) or number_of_phases < 1:
                print("Error: Number of phases should be an integer and not less than 1.")
                display("Error: Number of phases should be an integer and not less than 1.")
                return

            # Validate the policy name
            if not policy_name.strip():
                print("Error: Policy name cannot be empty.")
                display("Error: Policy name cannot be empty. Please provide a valid policy name.")
                return

            # Check if the policy name already exists in Policy_Data
            if policy_name in Policy_Data:
                display(f"NOTE: YOU WILL OVERRIDE THE EXISTING '{policy_name}' POLICY PREVIOUSLY CREATED IF CLICK ADD.")

            # Check if the number of phases has changed
            if current_policy_data["number_of_phase"] is not None and number_of_phases != current_policy_data["number_of_phase"]:
                display("Number of phases has changed. Resetting the policy configuration.")
                current_policy_data["out3"] = None  # Clear the current configuration state

            # Update the policy data
            current_policy_data["policy_name"] = policy_name
            current_policy_data["number_of_phase"] = number_of_phases

            try:
                # Generate a new configuration UI if needed
                if current_policy_data["out3"] is None:
                    #print("Creating new policy configuration")
                    out3 = config_user_policy(number_of_phases)
                    current_policy_data["out3"] = out3
                else:
                    #print("Using existing policy configuration")
                    out3 = current_policy_data["out3"]

                # Generate the correct message based on the number of phases
                phase_text = "phase" if number_of_phases == 1 else "phases"

                # Display the generated policy configuration UI
                #display(f"Policy '{policy_name}' defined with {number_of_phases} {phase_text}.")
                display(out3)

                # Show the "Update Policy" button
                update_button.layout.display = 'inline-block'

            except Exception as e:
                print(f"Error in on_define: {e}")
                display(f"Error in defining policy: {e}")

    def on_update(_):
        with output_display:
            output_display.clear_output()

            # Process and display the updated policy
            data_array = process_policy(current_policy_data["number_of_phase"], current_policy_data["out3"])

            if data_array is None:
                # Validation failed, stop processing
                return  # Error message is already displayed in process_policy()

            current_policy_data["data_array"] = data_array  # Update data array

            as_dictionary_for_display = return_for_multiple_phases(data_array)
            phase_keys = list(as_dictionary_for_display.keys())

            if phase_keys[0] == 'Dry':
                display(as_dictionary_for_display)
            else:
                for tmp_key in phase_keys:
                    display(f'Phase: {tmp_key}')
                    display(as_dictionary_for_display[tmp_key])

            # Temporarily also showing as DataArray
            display(data_array.attrs)

            # Show the "Add" button
            add_button.layout.display = 'inline-block'

    # Function to handle "Add" button click
    def on_add(_):
        with output_display:
            output_display.clear_output()
            policy_name = current_policy_data["policy_name"]
            data_array = current_policy_data["data_array"]

            if policy_name and data_array is not None:
                # Add the data array to Policy_Data
                Policy_Data[policy_name] = data_array
                display(f"Policy '{policy_name}' added to Pre-Transition Policies")
            else:
                display("Error: Policy name or data array is missing.")

    # Attach button click handlers
    define_button.on_click(on_define)
    update_button.on_click(on_update)
    add_button.on_click(on_add)

    # Display the widgets
    display(widgets.VBox([num_phase, pol_name, define_button, update_button, add_button, output_display]))
    
    
#####################

# Main function to create a new policy
def create_user_PT_policy(Post_Transition_Policy_Data):
    
    ########## HARD-CODED TO 1 PHASE FOR POST-TRANSITION ######
    # Set the number of phases for the user-defined policy
    num_phase = widgets.IntText(
        value=1,  # Fixed to 1 phase
        description='Number of phases:',
        style={'description_width': 'initial'},
        disabled=True  # Disable editing
    )

    # Set the name for the user-defined policy
    pol_name = widgets.Text(
        value='',
        description='Name of policy:',
        style={'description_width': 'initial'},
        disabled=False
    )

    # Buttons for actions
    define_button = widgets.Button(description="1) Define Policy")  # Updated button label
    update_button = widgets.Button(description="2) Review Policy")
    add_button = widgets.Button(description="3) Add Policy")  # New "Add" button
    define_button.layout.display = '230px'
    update_button.layout.display = '230px'  # Initially hidden
    add_button.layout.display = '230px'  # Initially hidden

    # Output widget to display results
    output_display = widgets.Output()

    # Track current policy configuration
    current_policy_data = {"number_of_phase": None, "policy_name": None, "out3": None, "data_array": None}


    def process_policy(number_of_phase, out3):
        """
        Process user-defined policy based on the widget structure from config_user_policy.
        Ensures water levels are in strictly increasing order (no duplicates).
        """
        user_dict = {}
        dyn_type = None
        phase_threshs = [] if number_of_phase > 1 else None

        try:
            # Iterate through each phase to extract water level thresholds and export amounts
            for i in range(number_of_phase):
                lev_dict = {}
                phase_widget = out3.children[i]  # Get the widget for the current phase

                # Extract water level thresholds
                thresh_widget = phase_widget.children[0].children[0]  # Water Level Thresholds text widget
                thresh_str = thresh_widget.value
                threshs = [int(item) for item in thresh_str.split()]  # Convert thresholds to integers

                # Validate that thresholds are strictly increasing (no duplicates)
                if threshs != sorted(set(threshs)):
                    display(f"Error: Phase {i + 1} water level thresholds are not strictly increasing. Please redefine.")
                    return None

                # Extract export amounts for each threshold
                export_tab = phase_widget.children[1].children[0]  # Tab widget for export amounts
                for j, thresh in enumerate(threshs):
                    ryt_values = []
                    for k in range(6):  # Assuming 6 year types
                        year_type_widget = export_tab.children[k]  # Text widget for each year type
                        value_str = year_type_widget.value
                        values = [int(item) for item in value_str.split()]
                        ryt_values.append(values[j])  # Collect export amounts for this threshold
                    lev_dict[thresh] = ryt_values

                user_dict[i + 1] = lev_dict

            # Extract dynamic type and phase thresholds if multiple phases
            if number_of_phase > 1:
                dyn_type = out3.children[-2].value  # Dynamic type dropdown widget
                phase_thresh_widget = out3.children[-1]  # Accordion for phase thresholds
                phase_threshs = [phase_thresh_widget.children[j].value for j in range(len(phase_thresh_widget.children))]

        except Exception as e:
            display(f"Error processing policy: {e}")
            return None

        # Create data array using the parsed user input
        return create_dataarray(user_dict, dyn_type, phase_threshs)

    def on_define(_):
        with output_display:
            output_display.clear_output()
            #print("Define button clicked")

            # Retrieve the current values from the widgets
            policy_name = pol_name.value
            number_of_phases = num_phase.value

            # Validate the number of phases
            if not isinstance(number_of_phases, int) or number_of_phases < 1:
                print("Error: Number of phases should be an integer and not less than 1.")
                display("Error: Number of phases should be an integer and not less than 1.")
                return

            # Validate the policy name
            if not policy_name.strip():
                print("Error: Policy name cannot be empty.")
                display("Error: Policy name cannot be empty. Please provide a valid policy name.")
                return

            # Check if the policy name already exists in Policy_Data
            if policy_name in Post_Transition_Policy_Data:
                display(f"NOTE: YOU WILL OVERRIDE THE EXISTING '{policy_name}' POLICY PREVIOUSLY CREATED IF CLICK ADD.")

            # Check if the number of phases has changed
            if current_policy_data["number_of_phase"] is not None and number_of_phases != current_policy_data["number_of_phase"]:
                display("Number of phases has changed. Resetting the policy configuration.")
                current_policy_data["out3"] = None  # Clear the current configuration state

            # Update the policy data
            current_policy_data["policy_name"] = policy_name
            current_policy_data["number_of_phase"] = number_of_phases

            try:
                # Generate a new configuration UI if needed
                if current_policy_data["out3"] is None:
                    #print("Creating new policy configuration")
                    out3 = config_user_policy(number_of_phases)
                    current_policy_data["out3"] = out3
                else:
                    #print("Using existing policy configuration")
                    out3 = current_policy_data["out3"]

                # Generate the correct message based on the number of phases
                phase_text = "phase" if number_of_phases == 1 else "phases"

                # Display the generated policy configuration UI
                #display(f"Policy '{policy_name}' defined with {number_of_phases} {phase_text}.")
                display(out3)

                # Show the "Update Policy" button
                update_button.layout.display = 'inline-block'

            except Exception as e:
                print(f"Error in on_define: {e}")
                display(f"Error in defining policy: {e}")

    def on_update(_):
        with output_display:
            output_display.clear_output()

            # Process and display the updated policy
            data_array = process_policy(current_policy_data["number_of_phase"], current_policy_data["out3"])

            if data_array is None:
                # Validation failed, stop processing
                return  # Error message is already displayed in process_policy()

            current_policy_data["data_array"] = data_array  # Update data array

            as_dictionary_for_display = return_for_multiple_phases(data_array)
            phase_keys = list(as_dictionary_for_display.keys())

            if phase_keys[0] == 'Dry':
                display(as_dictionary_for_display)
            else:
                for tmp_key in phase_keys:
                    display(f'Phase: {tmp_key}')
                    display(as_dictionary_for_display[tmp_key])

            # Temporarily also showing as DataArray
            ### do not need to show attributes for post-transition since just 1 phase
            #display(data_array.attrs)

            # Show the "Add" button
            add_button.layout.display = 'inline-block'

    # Function to handle "Add" button click
    def on_add(_):
        with output_display:
            output_display.clear_output()
            policy_name = current_policy_data["policy_name"]
            data_array = current_policy_data["data_array"]

            if policy_name and data_array is not None:
                # Add the data array to Policy_Data
                Post_Transition_Policy_Data[policy_name] = data_array
                display(f"Policy '{policy_name}' added to Post-Transition Policies")
            else:
                display("Error: Policy name or data array is missing.")

    # Attach button click handlers
    define_button.on_click(on_define)
    update_button.on_click(on_update)
    add_button.on_click(on_add)

    # Display the widgets
    display(widgets.VBox([num_phase, pol_name, define_button, update_button, add_button, output_display]))

    
    
    
    
#############################

from IPython.display import clear_output

def define_model_conditions_interface(Policy_Data, Post_Transition_Policy_Data):
    """
    Function to create a policy interface for managing initial water levels, 
    year selection (Wrapped or Projections), and policy configurations.
    """
    # Shared state dictionary to store all user selections
    state = {
        "Initial_Water_Level": 6383.7,
        "Wrapped_or_Projections": "Projections",
        "key_word": None,
        "Start_Year": None,
        "End_Year": None,
        "SSPs_of_Interest": [],
        "Pre_Transition_Policies": [],
        "Post_Transition_Policies": [],
    }

    ############################
    # Initial Water Level
    def create_initial_water_level_widget():
        description = widgets.HTML(
            "<b style='font-size:16px;'>Initial Water Level:</b>"
        )

        ini_water_lev = widgets.BoundedFloatText(
            value=state["Initial_Water_Level"],
            min=6300,
            max=6500,
            step=0.1,
            description='',  # No description in the widget itself
            style={'description_width': 'initial'},
            disabled=False,
            continuous_update=True
        )

        out1 = widgets.Output()

        with out1:
            print(f"Initial Water Level set to: {state['Initial_Water_Level']}")

        def on_value_change(change):
            state["Initial_Water_Level"] = change['new']
            with out1:
                clear_output()
                print(f"Initial Water Level set to: {state['Initial_Water_Level']}")

        ini_water_lev.observe(on_value_change, names='value')
        display(widgets.VBox([description, ini_water_lev, out1]))

    ############################
    # Year Selector (Wrapped or Projections)
    year_selector_output = widgets.Output()

    def set_attributes(Wrapped_or_Projections):
        if Wrapped_or_Projections == 'Wrapped':
            key_word = 'Sequence'
            min_start_year = 1955
            default_start_year = 1971
            max_end_year = 2020
        elif Wrapped_or_Projections == 'Projections':
            key_word = 'GCM'
            min_start_year = 1955
            default_start_year = 2024
            max_end_year = 2099
        else:
            raise ValueError("Invalid Wrapped_or_Projections value")
        return key_word, default_start_year, min_start_year, max_end_year

    def create_year_selector(Wrapped_or_Projections):
        state["key_word"], default_start_year, min_start_year, max_end_year = set_attributes(Wrapped_or_Projections)
        state["Start_Year"] = default_start_year
        state["End_Year"] = max_end_year

        with year_selector_output:
            clear_output(wait=True)
            if Wrapped_or_Projections == 'Wrapped':
                title = widgets.HTML("<b style='font-size:16px;'>Select Start and End Year for Wrapped Runs:</b>")
                subtitle = widgets.HTML(f"<p style='font-size:14px; color:gray;'>Years available: {min_start_year}-{max_end_year}, where end year predicts following years April 1st water level</p>")
            else:
                title = widgets.HTML("<b style='font-size:16px;'>Select Start and End Year for Projections:</b>")
                subtitle = widgets.HTML(f"<p style='font-size:14px; color:gray;'>Years available: {min_start_year}-{max_end_year}, where end year predicts following years April 1st water level</p>")

            start_year_widget = widgets.BoundedIntText(
                value=default_start_year,
                min=min_start_year,
                max=max_end_year - 2,
                step=1,
                description="Start Year:",
                style={'description_width': 'initial'}
            )
            
            end_year_widget = widgets.BoundedIntText(
                value=max_end_year,
                min=min_start_year + 1,
                max=max_end_year,
                step=1,
                description="End Year:",
                style={'description_width': 'initial'}
            )
            
            output = widgets.Output()

            def update_output(change=None):
                with output:
                    clear_output()
                    if start_year_widget.value >= end_year_widget.value:
                        print("Error: Start Year must be less than End Year.")
                    else:
                        state["Start_Year"] = start_year_widget.value
                        state["End_Year"] = end_year_widget.value
                        print(f"Start Year: {state['Start_Year']}")
                        print(f"End Year: {state['End_Year']}")

            start_year_widget.observe(update_output, names='value')
            end_year_widget.observe(update_output, names='value')

            display(widgets.VBox([title, subtitle, start_year_widget, end_year_widget, output]))

    def update_year_selector(change):
        state["Wrapped_or_Projections"] = change['new']
        create_year_selector(state["Wrapped_or_Projections"])

    wrap_or_proj = widgets.Select(
        options=['Wrapped', 'Projections'],
        value=state["Wrapped_or_Projections"],
        description='',
        rows=3,
        continuous_update=True
    )
    wrap_or_proj.observe(update_year_selector, names='value')

    ############################
    # SSP Selector
    def display_ssp_selector():
        ssp_options = ['ssp245', 'ssp370', 'ssp585']
        ssp_selector, SSPs_of_Interest = create_policy_selector(ssp_options, "Emission Scenario Options (only relevant for projections)", default_select_all=True)
        state["SSPs_of_Interest"] = SSPs_of_Interest
        display(ssp_selector)

    ############################
    # Policy Selectors
    def display_policy_selectors():
        pre_transition_policies = list(Policy_Data.keys())
        pre_transition_selector, Policy_list = create_policy_selector(pre_transition_policies, "Pre-Transition Policies")
        state["Pre_Transition_Policies"] = Policy_list
        display(pre_transition_selector)

        post_transition_policies = list(Post_Transition_Policy_Data.keys())
        post_transition_selector, Post_Transition_Policy_List = create_policy_selector(post_transition_policies, "Post-Transition Policies")
        state["Post_Transition_Policies"] = Post_Transition_Policy_List
        display(post_transition_selector)

    ############################
    # Display All Widgets
    create_initial_water_level_widget()
    description = widgets.HTML("<b style='font-size:16px;'>Wrapped or Projections:</b>")
    display(widgets.VBox([description, wrap_or_proj, year_selector_output]))
    create_year_selector(state["Wrapped_or_Projections"])
    display_ssp_selector()
    display_policy_selectors()

    return state









################################ FOR PLOTS

# PLOT 2

def create_checkbox_selector(options, title_text="Select Options", default_select_all=False):
    """
    Create a reusable checkbox-based selector with 'Select All' and 'Deselect All' buttons.

    Args:
        options (list): List of options to display as checkboxes.
        title_text (str): Title for the checkbox selector.
        default_select_all (bool): Whether all checkboxes should be selected by default.

    Returns:
        tuple: A tuple containing the VBox widget and a callable function to retrieve selected options.
    """
    # Create checkboxes
    checkboxes = [
        widgets.Checkbox(
            value=default_select_all,
            description=option,
            layout=widgets.Layout(margin='0 0 0 5px', width='auto')
        ) for option in options
    ]

    # Function to retrieve selected options
    def get_selected_options():
        return [cb.description for cb in checkboxes if cb.value]

    # Output widget
    output = widgets.Output()
    with output:
        print(f"Selected {title_text}: {get_selected_options()}")

    # Update output when a checkbox changes
    def on_checkbox_change(_):
        with output:
            output.clear_output()
            print(f"Selected {title_text}: {get_selected_options()}")

    for cb in checkboxes:
        cb.observe(on_checkbox_change, names='value')

    # Buttons to select and deselect all
    select_all_button = widgets.Button(description="Select All")
    deselect_all_button = widgets.Button(description="Deselect All")

    def select_all(_):
        for cb in checkboxes:
            cb.value = True

    def deselect_all(_):
        for cb in checkboxes:
            cb.value = False

    select_all_button.on_click(select_all)
    deselect_all_button.on_click(deselect_all)

    # Group widgets
    title = widgets.HTML(f"<b style='font-size:16px;'>{title_text}</b>")
    checkbox_group = widgets.VBox(
        checkboxes,
        layout=widgets.Layout(border='1px solid black', padding='5px', width='275px', align_items='flex-start')
    )
    selector_widget = widgets.VBox([title, widgets.HBox([select_all_button, deselect_all_button]), checkbox_group, output])

    return selector_widget, get_selected_options


def wrapper_plot_variable_time_series(
    Wrapped_or_Projections,
    mlib_plot_function,
    get_selected_policies,
    get_selected_pt_policies,
    **kwargs
):
    """
    Wrapper function to dynamically retrieve selected policies and pass them to a plotting function.

    Args:
        Wrapped_or_Projections: The projection data or wrapper object.
        mlib_plot_function (callable): The plotting function from the library.
        get_selected_policies (callable): Function to retrieve selected primary policies.
        get_selected_pt_policies (callable): Function to retrieve selected post-transition policies.
        **kwargs: Additional keyword arguments for the plotting function.
    """
    selected_policies = get_selected_policies()  # Retrieve selected primary policies
    selected_pt_policies = get_selected_pt_policies()  # Retrieve selected post-transition policies

    # Call the plotting function
    mlib_plot_function(
        Wrapped_or_Projections=Wrapped_or_Projections,
        policies=selected_policies,
        pt_policies_to_plot=selected_pt_policies,
        Variable=kwargs.get("Variable"),
        SSP_of_Interest=kwargs.get("SSP_of_Interest"),
        Policy_Outputs=kwargs.get("Policy_Outputs"),
        key_word=kwargs.get("key_word"),
        Start_Year=kwargs.get("Start_Year"),
        End_Year=kwargs.get("End_Year"),
        Sequence_of_Interest=kwargs.get("Sequence_of_Interest")
    )
    
def save_results_to_csv(path_to_save_to,name_of_file,Wrapped_or_Projections,Policy_list,Post_Transition_Policy_List,SSPs_of_Interest,Policy_Outputs):

    for pre_policy in Policy_list:

        for post_policy in Post_Transition_Policy_List:

            tmp = Policy_Outputs.sel(policy=pre_policy,PT_policy=post_policy)

            if Wrapped_or_Projections == 'Projections':

                for ssp in SSPs_of_Interest:

                    tmp_tmp = tmp.sel(ssp=ssp)

                    df = tmp_tmp.to_dataframe()

                    df_reset = df.reset_index()

                    # Step 2: Create a unique identifier for each GCM and SSP combination
                    df_reset['GCM_SSP'] = df_reset['GCM'] + '_' + df_reset['ssp']
                    column_name = 'GCM_SSP'           

                    # Step 3: Pivot the DataFrame to align GCMs side-by-side, excluding 'Transition_Time'
                    df_pivot = df_reset.pivot_table(
                        index=["year", "policy", "PT_policy"],  # Rows to keep
                        columns=column_name,  # Columns to stack side-by-side (GCM_SSP as unique identifier)
                        values=["Water_Level", "Storage", "Exports"],  # Values to display
                        aggfunc='first'  # Ensure the first non-NaN value is kept in case of duplicates
                    )

                    # Step 4: Flatten MultiIndex columns (optional for readability)
                    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

                    # Step 5: Reset index if desired
                    df_pivot.reset_index(inplace=True)

                    # Save to a CSV file
                    #df_pivot.to_csv(f'{path_to_save_to}/{pre_policy}_{post_policy}_{ssp}.csv')
                    
                    # Save to a excel file
                    file_path = os.path.join(path_to_save_to, f'{name_of_file}.xlsx')
                    if os.path.isfile(file_path):
                        with pd.ExcelWriter(file_path, mode='a') as writer:
                            df_pivot.to_excel(writer, sheet_name=f'{pre_policy}_{post_policy}_{ssp}', index=False)
                    else:
                        with pd.ExcelWriter(file_path, mode='w') as writer:
                            df_pivot.to_excel(writer, sheet_name=f'{pre_policy}_{post_policy}_{ssp}', index=False)

            else:

                df = tmp.to_dataframe()

                df_reset = df.reset_index()

                column_name = 'Sequence'                

                # Step 3: Pivot the DataFrame to align GCMs side-by-side, excluding 'Transition_Time'
                df_pivot = df_reset.pivot_table(
                    index=["year", "policy", "PT_policy"],  # Rows to keep
                    columns=column_name,  # Columns to stack side-by-side (GCM_SSP as unique identifier)
                    values=["Water_Level", "Storage", "Exports"],  # Values to display
                    aggfunc='first'  # Ensure the first non-NaN value is kept in case of duplicates
                )

                # Step 5: Reset index if desired
                df_pivot.reset_index(inplace=True)

                # Save to a CSV file
                #df_pivot.to_csv(f'{path_to_save_to}/{pre_policy}_{post_policy}.csv')
                
                # Save to a excel file
                file_path = os.path.join(path_to_save_to,f'{name_of_file}.xlsx')
                if os.path.isfile(file_path):
                    with pd.ExcelWriter(file_path, mode='a') as writer:
                        df_pivot.to_excel(writer, sheet_name=f'{pre_policy}_{post_policy}_{ssp}', index=False)
                else:
                    with pd.ExcelWriter(file_path, mode='w') as writer:
                        df_pivot.to_excel(writer, sheet_name=f'{pre_policy}_{post_policy}_{ssp}', index=False)

    print(f'Done saving results to {file_path}')
    
def get_data_path():
    gcm_data_path = input("Please enter the path of GCM data directory: ").strip()
    model_file = input("Please enter the path of model file: ").strip()
    lake_file = input("Please enter the path of lake file: ").strip()
    return (gcm_data_path, model_file, lake_file)
    
def write_data_to_excel(state,Policy_Outputs):
    Wrapped_or_Projections = state["Wrapped_or_Projections"]
    Policy_list = state["Pre_Transition_Policies"]
    Post_Transition_Policy_List = state["Post_Transition_Policies"]
    SSPs_of_Interest = state["SSPs_of_Interest"]
    user_input = input("Enter 1 to write results to excel file OR anything else not to write output")
    if user_input == '1':
        name_of_file = input("Enter name of file that will be written (e.g. policy_outputs): ").strip()
        path_to_save_to = ''
        save_results_to_csv(path_to_save_to,name_of_file,Wrapped_or_Projections,Policy_list,Post_Transition_Policy_List,SSPs_of_Interest,Policy_Outputs)
    else:
        pass

    
def plot_percent_in_transition_phase(Wrapped_or_Projections, policies_to_plot, ssps_to_plot, pt_policy, 
                                               Policy_Outputs, Start_Year, End_Year):
    
    ### Percent of GCMs or Sequences that Transition
    
    ### Just Plot % GCMs that Transition

    ### Just plot selected pairs of policies

    ### Single Figure Plot for A2 + None, A1 + D-1631, and A6 + PT3
    ### Percent of GCMs above water level for different SSPs

    # ####### USER-DEFINED ##########
    # ssps_to_plot = ['ssp245', 'ssp370', 'ssp585']  # List of SSPs
    # policies_to_plot = ['A1','A2','A6']#'A6']
    # ####### USER-DEFINED ##########

    Variable = 'Water_Level'
    pt_policies = [pt_policy for _ in range(len(policies_to_plot))]

    # Calculate years since Start_Year
    years_since_start = abs(Start_Year - np.arange(Start_Year, End_Year + 2))

    # Unique colors for each policy
    #colors = ['black', 'blue', 'green','orange']  # Assigning colors to A2, A1, and A6
    colors = cm.get_cmap('copper_r', len(policies_to_plot))

    if Wrapped_or_Projections == 'Projections':

        # Create subplots (rows = water levels, columns = SSPs)
        fig, axes = plt.subplots(1, len(ssps_to_plot), 
                                 figsize=(20, 7), 
                                 sharey=False)
        
        x_years = list(Policy_Outputs['year'].values)

        for col_idx, tmp_ssp in enumerate(ssps_to_plot):

            ax = axes[col_idx]

            for policy_idx, (policy, pt_policy) in enumerate(zip(policies_to_plot, pt_policies)):
                
                # Select the corresponding data for the policy and SSP
                tmp_array_ssp = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy=policy)

                ### get percent of GCMs that transition over time
                num_sequences = len(Policy_Outputs.GCM)
                perc_gcms_transition_over_time = []
                for year in range(0,(End_Year-Start_Year)+2):
                    perc_tmp = 100*(np.sum((tmp_array_ssp['Transition_Time']<year).values) / num_sequences)
                    perc_gcms_transition_over_time.append(perc_tmp)

                perc_gcms_transition_over_time = pd.Series(perc_gcms_transition_over_time,index=np.arange(Start_Year,End_Year+2))

                # Plot the data
                ax.plot(
                    perc_gcms_transition_over_time.index,
                    perc_gcms_transition_over_time,
                    label=f'{policy}' if col_idx == 0 else '',
                    color=colors(policy_idx),
                    linestyle='-' if 'A' in policy else '-',  # Different linestyle for specific policies
                    linewidth=2.5
                )

            # Customize the subplot
            ax.set_ylabel('Percent (%)' if col_idx == 0 else '', 
                          fontsize=24, labelpad=20)

            ax.set_ylim(0, 100)
            ax.grid()

            # Set x-axis ticks and labels
            x_yrs = x_years[::10]
            x_yrs.append(End_Year+1)
            ax.set_xticks(x_yrs)
            ax.set_xticklabels(x_yrs, rotation=45, fontsize=18)
        #         ax.set_yticklabels(ax.get_yticks(), fontsize=22)
            ax.set_yticks(range(0, 101, 20))  # Use integer ticks from 0 to 100 in increments of 20
            ax.set_yticklabels(range(0, 101, 20), fontsize=22)  # Set the labels to integers
            ax.set_ylim(0,105)

            ax.set_title(f"{tmp_ssp.upper()}", fontsize=25, fontweight='bold')
            ax.set_xlim(Start_Year, End_Year+1)

            # Add secondary x-axis for "Time since start of simulation (Years)"
            secax = ax.secondary_xaxis(-0.3)
            GCM_Range_of_Years = np.arange(Start_Year,End_Year+2)
            time_since_start = np.arange(0, len(GCM_Range_of_Years))
            step = 10  # Step size for labeling every 5 years
            xticks = np.append(GCM_Range_of_Years[::step], GCM_Range_of_Years[-1])
            xtick_labels = np.append(time_since_start[::step], len(GCM_Range_of_Years)-1)
            secax.set_xticks(xticks)
            secax.set_xticklabels(xtick_labels,fontsize=20)
            secax.set_xlabel('Time since start of Sequence (Yrs)', fontsize=22)


        # Add a common title for the figure
        fig.suptitle(f"Percent of GCMs Reached Post-Transition Phase", fontsize=28, fontweight='bold', y=1.035)

        # Add a legend above the figure
        fig.legend(
            labels=[f'{policy}' for policy in policies_to_plot],
            loc='upper center',
            bbox_to_anchor=(0.5, 0.98),
            ncol=len(policies_to_plot),
            fontsize=22
        )

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    else:

        # Create subplots (rows = water levels, columns = SSPs)
        fig, ax = plt.subplots(1, 1, 
                                 figsize=(7, 7), 
                                 sharey=False)

        for policy_idx, (policy, pt_policy) in enumerate(zip(policies_to_plot, pt_policies)):
            # Select the corresponding data for the policy and SSP
            tmp_array_ssp = Policy_Outputs.sel(PT_policy=pt_policy, policy=policy)

            num_sequences = len(Policy_Outputs.Sequence)

            ### get percent of GCMs that transition over time
            perc_gcms_transition_over_time = []
            for year in range(0,(End_Year-Start_Year)+2):
                perc_tmp = 100*(np.sum((tmp_array_ssp['Transition_Time']<year).values) / num_sequences)
                perc_gcms_transition_over_time.append(perc_tmp)

            perc_gcms_transition_over_time = pd.Series(perc_gcms_transition_over_time,index=np.arange(Start_Year,End_Year+2))

            # Plot the data
            ax.plot(
                perc_gcms_transition_over_time.index,
                perc_gcms_transition_over_time,
                label=f'{policy}',
                color=colors(policy_idx),
                linestyle='-' if 'A' in policy else '-',  # Different linestyle for specific policies
                linewidth=2.5
            )

        # Customize the subplot
        ax.set_ylabel('Percent (%)', 
                      fontsize=24, labelpad=20)

        ax.set_ylim(0, 100)
        ax.grid()

        # Set x-axis ticks and labels
        x_years = list(Policy_Outputs['year'].values)
        x_yrs = x_years[::10]
        x_yrs = [x + Start_Year for x in x_yrs]
        #x_yrs.append(End_Year+1)
        ax.set_xticks(x_yrs)
        ax.set_xticklabels(x_yrs, rotation=45, fontsize=18)
    #         ax.set_yticklabels(ax.get_yticks(), fontsize=22)
        ax.set_yticks(range(0, 101, 20))  # Use integer ticks from 0 to 100 in increments of 20
        ax.set_yticklabels(range(0, 101, 20), fontsize=22)  # Set the labels to integers
        ax.set_ylim(0,105)

        #ax.set_title(f"{tmp_ssp.upper()}", fontsize=25, fontweight='bold')
        ax.set_xlim(Start_Year, End_Year+1)

        # Add secondary x-axis for "Time since start of simulation (Years)"
        secax = ax.secondary_xaxis(-0.3)
        GCM_Range_of_Years = np.arange(Start_Year,End_Year+2)
        time_since_start = np.arange(0, len(GCM_Range_of_Years))
        step = 10  # Step size for labeling every 5 years
        xticks = np.append(GCM_Range_of_Years[::step], GCM_Range_of_Years[-1])
        xtick_labels = np.append(time_since_start[::step], len(GCM_Range_of_Years)-1)
        secax.set_xticks(xticks)
        secax.set_xticklabels(xtick_labels,fontsize=20)
        secax.set_xlabel('Time since start of Sequence (Yrs)', fontsize=22)


        # Add a common title for the figure
        fig.suptitle(f"Percent of Sequences Reached Post-Transition Phase", fontsize=28, fontweight='bold', y=1.035)

        # Add a legend above the figure
        fig.legend(
            labels=[f'{policy}' for policy in policies_to_plot],
            loc='upper center',
            bbox_to_anchor=(0.5, 0.98),
            ncol=len(policies_to_plot),
            fontsize=22
        )

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])    

# Create widgets
def create_widgets(state, Policy_Outputs):
    
    key_word = state["key_word"]
    Wrapped_or_Projections = state["Wrapped_or_Projections"]
    Policy_list = state["Pre_Transition_Policies"]
    Post_Transition_Policy_List = state["Post_Transition_Policies"]

    ### Single Choice

    Policy_of_Interest = widgets.Select(
        options = Policy_list,
        value = Policy_list[0],
        description = 'Policy of Interest:',
        style = {'description_width': 'initial'},
        rows = len(list(Policy_Outputs.policy.values)),
        continuous_update = True,
        disabled = False
    )

    PT_Policy_of_Interest = widgets.Select(
        options = Post_Transition_Policy_List,
        value = Post_Transition_Policy_List[0],
        description = 'Post Transition Policy of Interest:',
        style = {'description_width': 'initial'},
        rows = len(list(Policy_Outputs.PT_policy.values)),
        continuous_update = True,
        disabled = False
    )

    Variable = widgets.Select(
        options = ['Water_Level','Storage','Exports'],
        value = 'Water_Level',
        description = 'Variable of Interest:',
        style = {'description_width': 'initial'},
        rows = 3,
        continuous_update = True,
        disabled = False
    )

    if key_word == 'GCM':
        SSP_of_Interest = widgets.Select(
        options = list(Policy_Outputs.ssp.values),
        value = 'ssp370',
        description = 'SSP of Interest:',
        style = {'description_width': 'initial'},
        rows = 3,
        continuous_update = True,
        disabled = False
        )
    else:
        SSP_of_Interest = fixed('ssp370')

    ##########
    if Wrapped_or_Projections == 'Wrapped':
        Sequences = Policy_Outputs.Sequence.values
    else:
        Sequences = Policy_Outputs.GCM.values

    Sequences = Sequences.tolist()

    Sequence_of_Interest = widgets.Select(
        options = Sequences,
        value = Sequences[0],
        description = 'Sequence of Interest',
        style = {'description_width': 'initial'},
        #rows = len(list(Policy_Outputs.PT_policy.values)),
        continuous_update = True,
        disabled = False
    )
    return (Policy_of_Interest, PT_Policy_of_Interest, Variable, SSP_of_Interest, Sequence_of_Interest)

# Create widget for year of interest
def get_year_of_interest(state):
    Wrapped_or_Projections = state["Wrapped_or_Projections"]
    Start_Year = state["Start_Year"]
    End_Year = state["End_Year"]
    if Wrapped_or_Projections == 'Projections':
        # Create the widget for year of interest
        year_of_interest = widgets.IntText(
            value=2080,
            description=f'Year of Interest ({Start_Year}-{End_Year}):',
            style={'description_width': 'initial'},
            continuous_update=True,
            disabled=False
        )
    else:
        # Create the widget for year of interest
        year_of_interest = widgets.IntText(
            value=40,
            description=f'Years into Simulation: ({0}-{50}):',
            style={'description_width': 'initial'},
            continuous_update=True,
            disabled=False
        )
    return year_of_interest

# Wrapper function to dynamically retrieve selected policies and pass to boxplot function
def wrapper_boxplot_variable_by_policy(
    Wrapped_or_Projections,
    get_selected_policies,
    Variable,
    SSP_of_Interest,
    Policy_Outputs,
    year_of_interest,
    get_selected_pt_policies
):
    """
    Wrapper function to dynamically retrieve selected pre- and post-transition policies
    and pass them to the boxplot function.
    """
    
    # Retrieve selected pre-transition policies
    selected_policies = get_selected_policies()
    #print("Wrapper Function - Selected Pre-Transition Policies:", selected_policies)

    # Retrieve selected post-transition policies
    selected_pt_policies = get_selected_pt_policies()
    #print("Wrapper Function - Selected Post-Transition Policies:", selected_pt_policies)

    # Check if any pre-transition policies are selected
    if not selected_policies:
        print("Error: No pre-transition policies selected.")
        return

    # Check if at least two post-transition policies are selected
    if len(selected_pt_policies) < 2:
        print("Error: Please select at least two post-transition policies.")
        return

    # Call the main boxplot function
    boxplot_variable_by_policy(
        Wrapped_or_Projections=Wrapped_or_Projections,
        policies=selected_policies,  # Pass dynamically retrieved policies
        pt_policies_to_plot=selected_pt_policies,
        Variable=Variable,
        year_of_interest=year_of_interest,
        SSP_of_Interest=SSP_of_Interest,
        Policy_Outputs=Policy_Outputs
    )
    
def wrapper_plot_variable_time_series_by_policy_mean(
    Wrapped_or_Projections,
    get_selected_policies,
    Variable,
    SSP_of_Interest,
    Policy_Outputs,
    key_word,
    Start_Year,
    End_Year,
    Stdev,
    Min_Max,
    get_selected_pt_policies
):
    """
    Wrapper function to dynamically retrieve selected pre- and post-transition policies
    and pass them to the plotting function.
    """
    # Retrieve selected pre-transition policies
    selected_policies = get_selected_policies()
    #print("Wrapper Function - Selected Pre-Transition Policies:", selected_policies)

    # Retrieve selected post-transition policies
    selected_pt_policies = get_selected_pt_policies()
    #print("Wrapper Function - Selected Post-Transition Policies:", selected_pt_policies)

    # Check if any policies are selected
    if not selected_policies:
        print("Error: No pre-transition policies selected.")
        return
    if not selected_pt_policies:
        print("Error: No post-transition policies selected.")
        return

    # Call the main plotting function
    plot_variable_time_series_by_policy_mean(
        Wrapped_or_Projections=Wrapped_or_Projections,
        policies=selected_policies,  # Pass dynamically retrieved policies
        pt_policies_to_plot=selected_pt_policies,
        Variable=Variable,
        SSP_of_Interest=SSP_of_Interest,
        Policy_Outputs=Policy_Outputs,
        key_word=key_word,
        Start_Year=Start_Year,
        End_Year=End_Year,
        Stdev=Stdev,
        Min_Max=Min_Max
    )

# Wrapper function to dynamically retrieve selected policies and pass to boxplot function
def wrapper_boxplot_variables_by_policies(
    Wrapped_or_Projections,
    get_selected_policies,
    get_selected_variables,
    SSP_of_Interest,
    Policy_Outputs,
    year_of_interest,
    get_selected_pt_policies,
    key_word,
    Start_Year,
    End_Year
):
    """
    Wrapper function to dynamically retrieve selected pre- and post-transition policies
    and variables, and pass them to the boxplot function for comparison.
    """
    # Retrieve selected pre-transition policies
    selected_policies = get_selected_policies()
    # print("Wrapper Function - Selected Pre-Transition Policies:", selected_policies)

    # Retrieve selected post-transition policies
    selected_pt_policies = get_selected_pt_policies()
    # print("Wrapper Function - Selected Post-Transition Policies:", selected_pt_policies)

    # Retrieve selected variables
    selected_variables = get_selected_variables()
    # print("Wrapper Function - Selected Variables:", selected_variables)

    # Check if at least two post-transition policies are selected
    if len(selected_pt_policies) < 2:
        print("Error: Please select at least two post-transition policies to compare.")
        return

    # Call the main boxplot function
    boxplot_variables_by_policies(
        Wrapped_or_Projections=Wrapped_or_Projections,
        policies_to_plot=selected_policies,  # Pass dynamically retrieved pre-transition policies
        pt_policies_to_plot=selected_pt_policies,  # Pass dynamically retrieved post-transition policies
        variables_to_plot=selected_variables,  # Use dynamically retrieved variables
        year_of_interest=year_of_interest,
        SSP_of_Interest=SSP_of_Interest,
        Policy_Outputs=Policy_Outputs,
        key_word=key_word,
        Start_Year=Start_Year,
        End_Year=End_Year
    )

# Create the widget for Water Level of Interest
def get_water_level_of_interest():
    Water_Level_of_Interest = widgets.IntText(
        value = 6391,
        description = 'Water Level of Interest:',
        style = {'description_width': 'initial'},
        continuous_update = True,
        disabled = False
    )
    return Water_Level_of_Interest

# Create the widget for mutiple water levels of interest
def get_water_levels_of_interest():
    Water_Levels_of_Interest = widgets.Text(
            description = 'Water Levels of Interest:',
            value = '6392 6391 6388',
            style = {'description_width': 'initial'},
            disabled = False,
            continuous_update = True
        )
    return Water_Levels_of_Interest

# Create the widget for SSPs of Interest
def get_ssps_of_interest():
    SSPs_of_Interest = widgets.SelectMultiple(
        value = ['ssp245'],
        options = ['ssp245', 'ssp370', 'ssp585'],
        rows = 3,
        description = 'SSPs of Interest (multiple choices):',
        style = {'description_width': 'initial'},
        continuous_update = True,
        disabled = False
    )
    return SSPs_of_Interest

# Wrapper function to dynamically retrieve selected policies and pass to the GCM plot function
def wrapper_plot_gcm_percent_above_level_by_policy(
    Wrapped_or_Projections,
    get_selected_policies,
    get_selected_pt_policies,
    Variable,
    Water_Level_of_Interest,
    SSP_of_Interest,
    Policy_Outputs,
    key_word,
    Start_Year,
    End_Year
):
    """
    Wrapper function to dynamically retrieve selected pre- and post-transition policies
    and pass them to the GCM percent above level plotting function.
    """
    # Retrieve selected pre-transition policies
    selected_policies = get_selected_policies()
    #print("Wrapper Function - Selected Pre-Transition Policies:", selected_policies)

    # Retrieve selected post-transition policies
    selected_pt_policies = get_selected_pt_policies()
    #print("Wrapper Function - Selected Post-Transition Policies:", selected_pt_policies)

    # Check if at least one pre-transition policy is selected
    if not selected_policies:
        print("Error: Please select at least one pre-transition policy.")
        return

    # Check if at least one post-transition policy is selected
    if not selected_pt_policies:
        print("Error: Please select at least one post-transition policy.")
        return

    # Call the main plotting function
    plot_gcm_percent_above_level_by_policy(
        Wrapped_or_Projections=Wrapped_or_Projections,
        policies_to_plot=selected_policies,  # Pass dynamically retrieved pre-transition policies
        pt_policies_to_plot=selected_pt_policies,  # Pass dynamically retrieved post-transition policies
        Variable=Variable,
        Water_Level_of_Interest=Water_Level_of_Interest,
        SSP_of_Interest=SSP_of_Interest,
        Policy_Outputs=Policy_Outputs,
        key_word=key_word,
        Start_Year=Start_Year,
        End_Year=End_Year
    )

# Wrapper function to dynamically retrieve selected policies and pass to the GCM plot function
def wrapper_plot_plot_percent_in_transition_phase(
    Wrapped_or_Projections,
    get_selected_policies,
    SSPs_of_Interest,
    Post_Transition_Policy_List,
    Policy_Outputs,
    Start_Year,
    End_Year
):
    """
    Wrapper function to dynamically retrieve selected pre- and post-transition policies
    and pass them to the GCM percent above level plotting function.
    """
    # Retrieve selected pre-transition policies
    selected_policies = get_selected_policies()
    #print("Wrapper Function - Selected Pre-Transition Policies:", selected_policies)

    # Retrieve selected post-transition policies
    #selected_pt_policies = get_selected_pt_policies()
    #print("Wrapper Function - Selected Post-Transition Policies:", selected_pt_policies)

    # Check if at least one pre-transition policy is selected
    if not selected_policies:
        print("Error: Please select at least one pre-transition policy.")
        return

    # Call the main plotting function
    plot_percent_in_transition_phase(
        Wrapped_or_Projections=Wrapped_or_Projections,
        policies_to_plot=selected_policies,  # Pass dynamically retrieved pre-transition policies
        ssps_to_plot=SSPs_of_Interest,
        pt_policy=Post_Transition_Policy_List[0],
        Policy_Outputs=Policy_Outputs,
        Start_Year=Start_Year,
        End_Year=End_Year
    )

# Wrapper function to dynamically retrieve selected policies and pass to the GCM plot function
def wrapper_plot_gcm_percent_above_level_by_ssp(
    Wrapped_or_Projections,
    get_selected_policies,
    Selected_PT_Policy,
    ssps_to_plot,
    Variable,
    Water_Level_of_Interest,
    Policy_Outputs,
    key_word,
    Start_Year,
    End_Year
):
    """
    Wrapper function to dynamically retrieve selected pre- and post-transition policies
    and pass them to the GCM percent above level plotting function.
    """
    if Wrapped_or_Projections != 'Projections':
        sys.exit('Figure only relevant for Projections, not wrapped runs.')
        
    # Retrieve selected pre-transition policies
    selected_policies = get_selected_policies()

    # Get selected post-transition policies
    selected_pt_policy = Selected_PT_Policy

    # Check if at least one pre-transition policy is selected
    if not selected_policies:
        print("Error: Please select at least one pre-transition policy.")
        return

    # Call the main plotting function
    plot_gcm_percent_above_level_by_ssp(
        Wrapped_or_Projections=Wrapped_or_Projections,
        policies_to_plot=selected_policies,  # Pass dynamically retrieved pre-transition policies
        ssps_to_plot=ssps_to_plot,  # Pass selected SSPs
        Variable=Variable,
        pt_policy=selected_pt_policy,  # Pass the single selected post-transition policy
        Water_Level_of_Interest=Water_Level_of_Interest,
        Policy_Outputs=Policy_Outputs,
        key_word=key_word,
        Start_Year=Start_Year,
        End_Year=End_Year
    )

# Function wrapper for the interactive widget
def wrapper_function(
    Wrapped_or_Projections,
    ssps_to_plot,
    Variable,
    pt_policy,
    Exports_Variable,
    Water_Level_of_Interest,
    Policy_Outputs,
    ryt_sef_xarray,
    key_word,
    Start_Year,
    End_Year,
    get_selected_policies
):
    """
    Wrapper function to dynamically retrieve selected pre-transition policies
    and pass them to the percent_diff_gcm_above_level_by_ssp_policy function.
    """
    if Wrapped_or_Projections != 'Projections':
        sys.exit('Figure Available for Projections, but not Wrapped Runs.')
    
    # Dynamically retrieve selected pre-transition policies
    policies_to_plot = get_selected_policies()  # Use the dynamically updated function
    print(f"Selected Pre-Transition Policies: {policies_to_plot}")  # Debug print

    # Check if any pre-transition policies are selected
    if not policies_to_plot:
        print("Error: Please select at least one pre-transition policy.")
        return

    # Call the main function with the dynamically selected policies
    percent_diff_gcm_above_level_by_ssp_policy(
        Wrapped_or_Projections=Wrapped_or_Projections,
        policies_to_plot=policies_to_plot,
        ssps_to_plot=ssps_to_plot,
        Variable=Variable,
        pt_policy=pt_policy,
        Exports_Variable=Exports_Variable,
        Water_Level_of_Interest=Water_Level_of_Interest,
        Policy_Outputs=Policy_Outputs,
        ryt_sef_xarray=ryt_sef_xarray,
        key_word=key_word,
        Start_Year=Start_Year,
        End_Year=End_Year
    )
    
### Only works if A1+D_1631 or A1+None run
### Bar plot of percent of GCMs at/above water level of interest (for each decade!)
### also includes average GCM dry and wet exports across each decade
def plot_percent_diff_pre_post_trans_with_export(
    Policy_Outputs,
    ssps_to_plot, 
    Water_Level_of_Interest, 
    policies_to_plot, 
    pt_policy, 
    Variable, 
    Exports_Variable, 
    Compare_to_A1_or_A2,
    Start_Year, 
    End_Year,
    ryt_sef_xarray):
    
    if Compare_to_A1_or_A2 == 'A1':
        pt_A1_A2 = 'D_1631'
    else:
        pt_A1_A2 = 'None'
    
    # Define the number of years in a decade (10 years)
    years_per_decade = 10
    # # Calculate years since Start_Year
    years_since_start = abs(Start_Year - np.arange(Start_Year, End_Year + 2))
    # Calculate number of decades
    num_decades = len(years_since_start) // years_per_decade

    # # Create a colormap with enough colors for all decades
    colormap = cm.get_cmap('copper_r', num_decades)

    # Update the grid size based on the number of policies
    num_policies = len(policies_to_plot)+1 ### add 1 to include baseline at beginning that comparing against
    grid_size = int(np.ceil(np.sqrt(num_policies)))  # Square grid layout

    # Create subplots for each policy in a grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 18))

    # Flatten axes to easily iterate
    axes = axes.flatten()

    # Bar width and spacing adjustments
    bar_width = 0.15  # Smaller width for more spacing between bars
    gap_width = 0.2   

    # Variables to track global min and max for export differences
    global_export_min = float('inf')
    global_export_max = float('-inf')

    # Special case for A1: Calculate export range across all decades for A1
    a1_export_min = float('inf')
    a1_export_max = float('-inf')

    for ssp_idx, tmp_ssp in enumerate(ssps_to_plot):

        a1_export_array = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_A1_A2, policy=Compare_to_A1_or_A2)[Exports_Variable]

        for decade in range(num_decades):
            start_idx = decade * years_per_decade
            end_idx = start_idx + years_per_decade

            # Calculate the slice of exports for the current decade
            a1_export_decade = a1_export_array.isel(year=slice(start_idx, end_idx))

            # Get the year types for the current SSP and decade
            tmp_ryt_array = ryt_sef_xarray.sel(ssp=tmp_ssp)
            tmp_ryt_array_decade = tmp_ryt_array.isel(year=slice(start_idx, end_idx))['Year_Type']

            # Apply masks for dry and wet year types
            dry_mask = np.isin(tmp_ryt_array_decade, dry_types)
            wet_mask = np.isin(tmp_ryt_array_decade, wet_types)

            # Compute average exports for dry and wet year types
            dry_exports = a1_export_decade.where(dry_mask)
            wet_exports = a1_export_decade.where(wet_mask)

            # Calculate dry and wet year min and max for A1
            dry_exports_mean = dry_exports.mean(dim=['GCM', 'year'], skipna=True).values
            wet_exports_mean = wet_exports.mean(dim=['GCM', 'year'], skipna=True).values

            # Update export min and max for A1
            a1_export_min = min(a1_export_min, dry_exports_mean, wet_exports_mean)
            a1_export_max = max(a1_export_max, dry_exports_mean, wet_exports_mean)

    # First pass to calculate global export min and max for other policies (excluding A1 for difference calculations)
    for policy_idx, tmp_policy in enumerate(policies_to_plot):  # Skip A1 for difference calculations

        for ssp_idx, tmp_ssp in enumerate(ssps_to_plot):

            export_policy = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy=tmp_policy)[Exports_Variable]
            export_baseline = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy='A1')[Exports_Variable]

            for decade in range(num_decades):

                start_idx = decade * years_per_decade
                end_idx = start_idx + years_per_decade

                # Extract the decade data for the policy and baseline (A1)
                export_policy_decade = export_policy.isel(year=slice(start_idx, end_idx))
                export_baseline_decade = export_baseline.isel(year=slice(start_idx, end_idx))

                # Get the year types for the current SSP and decade
                tmp_ryt_array = ryt_sef_xarray.sel(ssp=tmp_ssp)
                tmp_ryt_array_decade = tmp_ryt_array.isel(year=slice(start_idx, end_idx))['Year_Type']

                # Apply masks for dry and wet year types
                dry_mask = np.isin(tmp_ryt_array_decade, dry_types)
                wet_mask = np.isin(tmp_ryt_array_decade, wet_types)

                # Compute export differences for dry and wet year types
                dry_export_policy = export_policy_decade.where(dry_mask)
                dry_export_baseline = export_baseline_decade.where(dry_mask)

                if Compare_to_A1_or_A2 == 'A1':
                    dry_export_diff = (dry_export_policy.mean(dim=['GCM', 'year'], skipna=True) - dry_export_baseline.mean(dim=['GCM', 'year'], skipna=True)).values
                else:
                    dry_export_diff = (dry_export_baseline.mean(dim=['GCM', 'year'], skipna=True) - dry_export_policy.mean(dim=['GCM', 'year'], skipna=True)).values

                wet_export_policy = export_policy_decade.where(wet_mask)
                wet_export_baseline = export_baseline_decade.where(wet_mask)

                if Compare_to_A1_or_A2 == 'A1':
                    wet_export_diff = (wet_export_policy.mean(dim=['GCM', 'year'], skipna=True) - wet_export_baseline.mean(dim=['GCM', 'year'], skipna=True)).values
                else:
                    wet_export_diff = (wet_export_baseline.mean(dim=['GCM', 'year'], skipna=True) - wet_export_policy.mean(dim=['GCM', 'year'], skipna=True)).values

                # Update global min and max for export differences based on dry and wet years
                global_export_min = min(global_export_min, dry_export_diff, wet_export_diff)
                global_export_max = max(global_export_max, dry_export_diff, wet_export_diff)

    # Round a1_export_max to the nearest 1000 and multiply by -1 for global_export_min
    a1_export_max = np.round(a1_export_max, -3)

    if Compare_to_A1_or_A2 == 'A1':
        global_export_min = -1 * a1_export_max
    else:
        global_export_min = -22000#-1 * global_export_max
        global_export_max = 0

    ###### First Plot what comparing against

    ax = axes[0]  # Select the correct subplot for each policy

    # Create secondary y-axis for exports
    if Compare_to_A1_or_A2 == 'A2':
        pass
    else:
        ax2 = ax.twinx()

    # Positions for each SSP on the x-axis, with increased gap
    positions = np.arange(len(ssps_to_plot)) * (bar_width * num_decades + gap_width)

    # Iterate over the SSPs and plot actual percent of GCMs and exports for A1
    for ssp_idx, tmp_ssp in enumerate(ssps_to_plot):
        a1_array_ssp = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_A1_A2, policy=Compare_to_A1_or_A2)[Variable]
        a1_export_array = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_A1_A2, policy=Compare_to_A1_or_A2)[Exports_Variable]

        # Decades' percent of GCMs at or above the water level and average exports for the current SSP
        a1_percent_above = []
        dry_exports_mean_list = []
        wet_exports_mean_list = []

        for decade in range(num_decades):
            # Slice data for the current decade
            start_idx = decade * years_per_decade
            end_idx = start_idx + years_per_decade
            a1_decade = a1_array_ssp.isel(year=slice(start_idx, end_idx))
            a1_export_decade = a1_export_array.isel(year=slice(start_idx, end_idx))

            # Calculate the percent of GCMs at or above the water level of interest
            percent_above_a1 = (a1_decade >= Water_Level_of_Interest).sum(dim='GCM') / len(a1_decade.GCM) * 100
            a1_percent_above.append(percent_above_a1.mean(dim='year').values)

            ################################# RYT ###############################
            # (Dry, Dry-Normal, Normal) vs > Normal (Wet-Normal, Wet, Extreme-Wet)
            ## get RYT for ssp of interest
            tmp_ryt_array = ryt_sef_xarray.sel(ssp=tmp_ssp)
            ## get slice of years of interest
            tmp_ryt_array_decade = tmp_ryt_array.isel(year=slice(start_idx, end_idx))
            ## intersted in year_type
            tmp_ryt_array_decade = tmp_ryt_array_decade['Year_Type']

            # Mask for "Dry", "Dry-Normal", and "Normal" year types
            dry_mask = np.isin(tmp_ryt_array_decade, dry_types)

            # Mask for "Wet-Normal", "Wet", and "Extreme-Wet" year types
            wet_mask = np.isin(tmp_ryt_array_decade, wet_types)

            # Compute average exports for "Dry", "Dry-Normal", and "Normal" year types
            dry_exports = a1_export_decade.where(dry_mask)  # Apply the mask
            dry_exports_mean = dry_exports.mean(dim=['GCM', 'year'], skipna=True)  # Compute the mean over GCM and year
            dry_exports_mean_list.append(dry_exports_mean.values)

            # Compute average exports for "Wet-Normal", "Wet", and "Extreme-Wet" year types
            wet_exports = a1_export_decade.where(wet_mask)  # Apply the mask
            wet_exports_mean = wet_exports.mean(dim=['GCM', 'year'], skipna=True)  # Compute the mean over GCM and year
            wet_exports_mean_list.append(wet_exports_mean.values)

        # Plot bars for the percent of GCMs at or above the water level
        for decade in range(num_decades):
            bar_pos = positions[ssp_idx] + decade * bar_width
            bar_height = a1_percent_above[decade]

            # Plot percent above water level bars
            ax.bar(bar_pos, bar_height, width=bar_width, label=f'Decade {decade + 1}' if ssp_idx == 0 else "", color=colormap(decade))

            # Add decade label above the bar
            ax.text(bar_pos, bar_height + 0.25, f'{decade + 1}', ha='center', va='bottom', fontsize=11)

        if Compare_to_A1_or_A2 == 'A2':
            pass
        else:
            # Plot dry_exports_mean and wet_exports_mean bars on the secondary y-axis
            for decade in range(num_decades):
                bar_pos = positions[ssp_idx] + decade * bar_width
                dry_export_mean = dry_exports_mean_list[decade]
                wet_export_mean = wet_exports_mean_list[decade]

                # Plot wet_exports_mean as blue
                if np.isfinite(wet_export_mean):
                    ax2.bar(bar_pos, wet_export_mean, width=bar_width, alpha=0.3, color='blue')

                # Plot dry_exports_mean as red with some alpha
                if np.isfinite(dry_export_mean):
                    ax2.bar(bar_pos, dry_export_mean, width=bar_width, alpha=0.5, edgecolor='red', fill=False, linewidth=2.0)

    #                 # Plot wet_exports_mean as blue, stacked on top of dry_exports_mean
    #                 if np.isfinite(wet_export_mean) and np.isfinite(dry_export_mean):
    #                     ax2.bar(bar_pos, wet_export_mean, bottom=dry_export_mean, width=bar_width, alpha=0.5, color='blue')
    #                 elif np.isfinite(wet_export_mean):
    #                     ax2.bar(bar_pos, wet_export_mean, width=bar_width, alpha=0.5, color='blue')
    positions = np.arange(len(ssps_to_plot)) * (bar_width * num_decades + gap_width)
    ax.set_xticks(positions + bar_width * (num_decades - 1) / 2)  # Center the tick labels
    ax.set_xticklabels(ssps_to_plot,fontsize=16)

    ax.text(-1.0,0,f'Percent GCMs Above {Water_Level_of_Interest} ft',rotation=90,fontsize=16,color='black')

    if Compare_to_A1_or_A2 == 'A2':
        ax.set_title('No Exports in\nPre- or Post-Transition',fontweight='bold',fontsize=18)
    elif Compare_to_A1_or_A2 == 'A1':
        ax.set_title('Existing Policy (A1)\nWith D-1631 Post-Transition',fontsize=18)

    if Compare_to_A1_or_A2 == 'A2':
        pass
    else:
        ax.text(4.5,10,f'Exports (ac-ft)',rotation=270,fontsize=16,color='grey')
        # Set the secondary y-axis limits for exports based on the range for A1
        ax2.set_ylim(0, a1_export_max)

    # Second pass to plot the data
    for policy_idx, tmp_policy in enumerate(policies_to_plot):

        actual_index = policy_idx + 1 ### to account for baseline comparing against

        ax = axes[actual_index]  # Select the correct subplot for each policy

        # Create secondary y-axis for exports
        ax2 = ax.twinx()

        # Positions for each SSP on the x-axis, with increased gap
        positions = np.arange(len(ssps_to_plot)) * (bar_width * num_decades + gap_width)

        # For other policies, calculate differences from A1 and percent difference
        max_bar_height_all_ssps = 0
        for ssp_idx, tmp_ssp in enumerate(ssps_to_plot):

            # Select the data for the policy and A1
            export_policy = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy=tmp_policy)[Exports_Variable]
            export_baseline = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_A1_A2, policy=Compare_to_A1_or_A2)[Exports_Variable]
            tmp_array_ssp = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy=tmp_policy)[Variable]
            baseline_array = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_A1_A2, policy=Compare_to_A1_or_A2)[Variable]

            dry_exports_diff_list = []
            wet_exports_diff_list = []
            decade_means = []

            for decade in range(num_decades):
                start_idx = decade * years_per_decade
                end_idx = start_idx + years_per_decade
                export_policy_decade = export_policy.isel(year=slice(start_idx, end_idx))
                export_baseline_decade = export_baseline.isel(year=slice(start_idx, end_idx))

                # Calculate percent difference in GCMs reaching the water level of interest
                tmp_policy_decade = tmp_array_ssp.isel(year=slice(start_idx, end_idx))
                baseline_decade = baseline_array.isel(year=slice(start_idx, end_idx))
                percent_above_policy = (tmp_policy_decade >= Water_Level_of_Interest).sum(dim='GCM') / len(tmp_policy_decade.GCM) * 100
                percent_above_baseline = (baseline_decade >= Water_Level_of_Interest).sum(dim='GCM') / len(baseline_decade.GCM) * 100

                if Compare_to_A1_or_A2 == 'A1':
                    percent_diff = (percent_above_policy - percent_above_baseline).mean(dim='year').values
                else:
                    percent_diff = (percent_above_baseline-percent_above_policy).mean(dim='year').values
                decade_means.append(percent_diff)

                # Get the year types for the current SSP and decade
                tmp_ryt_array = ryt_sef_xarray.sel(ssp=tmp_ssp)
                tmp_ryt_array_decade = tmp_ryt_array.isel(year=slice(start_idx, end_idx))['Year_Type']

                # Apply masks for dry and wet year types
                dry_mask = np.isin(tmp_ryt_array_decade, dry_types)
                wet_mask = np.isin(tmp_ryt_array_decade, wet_types)

                # Compute the export differences for dry and wet year types relative to A1
                dry_export_policy = export_policy_decade.where(dry_mask)
                dry_export_baseline = export_baseline_decade.where(dry_mask)

                if Compare_to_A1_or_A2 == 'A1':
                    dry_export_diff = (dry_export_policy.mean(dim=['GCM', 'year']) - dry_export_baseline.mean(dim=['GCM', 'year']))
                else:
                    dry_export_diff = (dry_export_baseline.mean(dim=['GCM', 'year']) - dry_export_policy.mean(dim=['GCM', 'year']))

                wet_export_policy = export_policy_decade.where(wet_mask)
                wet_export_baseline = export_baseline_decade.where(wet_mask)

                if Compare_to_A1_or_A2 == 'A1':
                    wet_export_diff = (wet_export_policy.mean(dim=['GCM', 'year']) - wet_export_baseline.mean(dim=['GCM', 'year']))
                else:
                    wet_export_diff = (wet_export_baseline.mean(dim=['GCM', 'year']) - wet_export_policy.mean(dim=['GCM', 'year']))

                dry_exports_diff_list.append(dry_export_diff.values)
                wet_exports_diff_list.append(wet_export_diff.values)

            # Plot percent difference bars for GCMs reaching the water level (primary y-axis)

            #print(f"Plotting policy {tmp_policy} in axes index {actual_index}")

            max_bar_height = np.ceil(np.max(decade_means) / 5) * 5 + 10

            if max_bar_height > max_bar_height_all_ssps:
                max_bar_height_all_ssps = max_bar_height

            for decade in range(num_decades):
                bar_pos = positions[ssp_idx] + decade * bar_width
                bar_height = decade_means[decade]

                ax.bar(bar_pos, bar_height, width=bar_width, label=f'Decade {decade + 1}' if ssp_idx == 0 else "", color=colormap(decade))

                # Add decade label above the bar
                ax.text(bar_pos, bar_height + 0.25, f'{decade + 1}', ha='center', va='bottom', fontsize=11)

            # Plot dry_exports_diff and wet_exports_diff bars on the secondary y-axis
            for decade in range(num_decades):
                bar_pos = positions[ssp_idx] + decade * bar_width
                dry_export_diff = dry_exports_diff_list[decade]
                wet_export_diff = wet_exports_diff_list[decade]

                # Plot dry_exports_diff as red outline
                if np.isfinite(dry_export_diff):
                    ax2.bar(bar_pos, dry_export_diff, width=bar_width, alpha=0.5, edgecolor='red', fill=False, linewidth=2.0)

                # Plot wet_exports_diff as blue
                if np.isfinite(wet_export_diff):
                    ax2.bar(bar_pos, wet_export_diff, width=bar_width, alpha=0.3, color='blue')

        ax.set_ylim(0,max_bar_height_all_ssps)

        # Customize secondary y-axis for exports
        ax2.set_ylim(global_export_min, global_export_max)  # Apply consistent y-limits across other subplots
        ax2.grid(False)

        # Add a baseline line for 0% change
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # Customize the subplot
        ax.set_xticks(positions + bar_width * (num_decades - 1) / 2)  # Center the tick labels
        ax.set_xticklabels(ssps_to_plot,fontsize=16)
        ax.set_title(f'{tmp_policy}', fontsize=18, fontweight='bold')
        ax.grid(axis='y')

        ax.set_ylim(0,60)

        if Compare_to_A1_or_A2 == 'A2':

            # Modify primary y-axis tick labels
            ticks = ax.get_yticks()  # Get current tick positions
            ax.set_yticklabels([f'-{int(abs(tick))}' for tick in ticks], fontsize=14)  # Add negative sign to the labels
            # Modify secondary y-axis tick labels
            ticks_secondary = ax2.get_yticks()  # Get current secondary tick positions
            ax2.set_yticklabels([f'{int(abs(tick))}' for tick in ticks_secondary], fontsize=14)  # Add negative sign to the labels

    # Hide unused subplots
    for i in range(len(policies_to_plot)+1, grid_size * grid_size):
        fig.delaxes(axes[i])

    # Set common titles and labels
    if Compare_to_A1_or_A2 == 'A2':
        fig.suptitle(f"Difference Relative to No Exports in Pre- and Post-Transition (A2+None)\nTop Left Plot = A2+None; All other plots = difference from A2+None", fontsize=25,fontweight='bold',y=1.005)
    elif Compare_to_A1_or_A2 == 'A1':
        fig.suptitle(f"Difference Relative to Existing Pre- and Post-Transition Policy (A1+D_1631)\nTop Left Plot = A1+D_1631; All other plots = difference from A1+D_1631",fontsize=25,fontweight='bold',y=1.005)

    if Compare_to_A1_or_A2 == 'A2':
        fig.text(-0.02, 0.5, f'Percent Difference in GCMs Above {Water_Level_of_Interest} ft', va='center', rotation='vertical', fontsize=24,fontweight='bold')
    elif Compare_to_A1_or_A2 == 'A1':
        fig.text(-0.02, 0.5, f'Percent Difference in GCMs Above {Water_Level_of_Interest} ft', va='center', rotation='vertical', fontsize=24,fontweight='bold')

    if Compare_to_A1_or_A2 == 'A2':
        fig.text(1.005, 0.5, f'Difference in Exports (ac-ft)', va='center', rotation=270, fontsize=24,fontweight='bold')
    elif Compare_to_A1_or_A2 == 'A1':
        fig.text(1.005, 0.5, f'Difference in Exports (ac-ft)', va='center', rotation=270, fontsize=24,fontweight='bold')
        
        
    custom_legend = [
        mpatches.Patch(color="peru", label="Percent Difference in Likelihood of Reaching Water Level (left y-axis)"),
        mpatches.Patch(color="lightsteelblue", label="Average Decadal Difference in Wet Year Exports (right y-axis)"),
        mpatches.Patch(facecolor='None',edgecolor="red", label="Average Decadal Difference in Dry Year Exports (right y-axis)"),
    ]

    # ✅ **Add the Custom Legend**
    fig.legend(handles=custom_legend, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=1, frameon=True,fontsize=16)

    # Adjust layout
    plt.tight_layout()

### Only works if A1+D_1631 or A1+None run
### Same as above but no exports!!!!
### Bar plot of percent of GCMs at/above water level of interest (for each decade!)
def plot_percent_diff_pre_post_trans_without_export(
    Policy_Outputs,
    ssps_to_plot, 
    Water_Level_of_Interest, 
    policies_to_plot, 
    pt_policy, 
    Variable, 
    Compare_to_A1_or_A2,
    Start_Year, End_Year):
    
    if Compare_to_A1_or_A2 == 'A1':
        pt_A1_A2 = 'D_1631'
    else:
        pt_A1_A2 = 'None'
    
    ### Define decade settings
    years_per_decade = 10
    years_since_start = abs(Start_Year - np.arange(Start_Year, End_Year + 2))
    num_decades = len(years_since_start) // years_per_decade

    ### Color settings
    colormap = cm.get_cmap('copper_r', num_decades)

    # Update grid size for subplots
    num_policies = len(policies_to_plot) + 1  # Add 1 to include baseline at the beginning
    grid_size = int(np.ceil(np.sqrt(num_policies)))

    # Create subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 18))
    axes = axes.flatten()

    # Bar width and spacing
    bar_width = 0.15
    gap_width = 0.2

    # First plot for the baseline comparison
    ax = axes[0]

    positions = np.arange(len(ssps_to_plot)) * (bar_width * num_decades + gap_width)

    for ssp_idx, tmp_ssp in enumerate(ssps_to_plot):
        a1_array_ssp = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_A1_A2, policy=Compare_to_A1_or_A2)[Variable]

        # Calculate the percentage above the water level
        a1_percent_above = []

        for decade in range(num_decades):
            start_idx = decade * years_per_decade
            end_idx = start_idx + years_per_decade
            a1_decade = a1_array_ssp.isel(year=slice(start_idx, end_idx))
            percent_above_a1 = (a1_decade >= Water_Level_of_Interest).sum(dim='GCM') / len(a1_decade.GCM) * 100
            a1_percent_above.append(percent_above_a1.mean(dim='year').values)

        # Plot bars
        for decade in range(num_decades):
            bar_pos = positions[ssp_idx] + decade * bar_width
            bar_height = a1_percent_above[decade]
            ax.bar(bar_pos, bar_height, width=bar_width, label=f'Decade {decade + 1}' if ssp_idx == 0 else "", color=colormap(decade))
            ax.text(bar_pos, bar_height + 0.25, f'{decade + 1}', ha='center', va='bottom', fontsize=14)

    ax.set_ylabel(f'Percent GCMs Above {Water_Level_of_Interest} ft', fontsize=18)
    ax.set_xticks(positions + bar_width * (num_decades - 1) / 2)
    ax.set_xticklabels(ssps_to_plot, fontsize=16)
    ax.set_title('No Exports\nPre- and Post-Transition', fontsize=20, fontweight='bold')
    ax.grid(axis='y')

    # Second pass for other policies
    for policy_idx, tmp_policy in enumerate(policies_to_plot):
        actual_index = policy_idx + 1
        ax = axes[actual_index]

        positions = np.arange(len(ssps_to_plot)) * (bar_width * num_decades + gap_width)
        for ssp_idx, tmp_ssp in enumerate(ssps_to_plot):
            tmp_array_ssp = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_policy, policy=tmp_policy)[Variable]
            baseline_array = Policy_Outputs.sel(ssp=tmp_ssp, PT_policy=pt_A1_A2, policy=Compare_to_A1_or_A2)[Variable]

            percent_diff_means = []

            for decade in range(num_decades):
                start_idx = decade * years_per_decade
                end_idx = start_idx + years_per_decade

                tmp_policy_decade = tmp_array_ssp.isel(year=slice(start_idx, end_idx))
                baseline_decade = baseline_array.isel(year=slice(start_idx, end_idx))
                percent_above_policy = (tmp_policy_decade >= Water_Level_of_Interest).sum(dim='GCM') / len(tmp_policy_decade.GCM) * 100
                percent_above_baseline = (baseline_decade >= Water_Level_of_Interest).sum(dim='GCM') / len(baseline_decade.GCM) * 100
                percent_diff = (percent_above_policy - percent_above_baseline).mean(dim='year').values
                percent_diff_means.append(percent_diff)

            # Plot bars
            for decade in range(num_decades):
                bar_pos = positions[ssp_idx] + decade * bar_width
                bar_height = percent_diff_means[decade]
                ax.bar(bar_pos, bar_height, width=bar_width, label=f'Decade {decade + 1}' if ssp_idx == 0 else "", color=colormap(decade))
                ax.text(bar_pos, bar_height + 0.25, f'{decade + 1}', ha='center', va='bottom', fontsize=14)

        #ax.set_ylabel(f'Percent Difference (%)', fontsize=18)
        ax.set_xticks(positions + bar_width * (num_decades - 1) / 2)
        ax.set_xticklabels(ssps_to_plot, fontsize=16)
        ax.set_title(f'{tmp_policy}', fontsize=20, fontweight='bold')
        ax.grid(axis='y')

        if pt_policy == 'D_1631':
            ax.set_ylim(0,-55)
        else:
            ax.set_ylim(0,-50)

    # Hide unused subplots
    for i in range(len(policies_to_plot) + 1, len(axes)):
        fig.delaxes(axes[i])

    # Common title and labels
    fig.suptitle(f"Percent Difference in GCMs Above {Water_Level_of_Interest} ft\nUsing Post-Transition: {pt_policy}", fontsize=24, fontweight='bold', y=1.02)
    #fig.text(0.5, -0.02, 'SSPs', va='center', ha='center', fontsize=18, fontweight='bold')
    fig.text(-0.03, 0.5, f'Percent Difference from No Exports in Pre/Post-Transition (%)', va='center', rotation='vertical', fontsize=24, fontweight='bold')

    # Adjust layout
    plt.tight_layout()
    
### Performance for Selected Metrics as y-coordinate plot
def line_plot_policy_performance(
    Policy_Outputs, 
    start_year_index, 
    end_year_index, 
    get_policies_to_plot, 
    ssps_to_plot, 
    get_pt_policies, 
    water_levels_of_interest_str, 
    Start_Year,
    ryt_sef_xarray):
    
    water_levels_of_interest = [int(item) for item in water_levels_of_interest_str.split()]
    print('Water levels of interest:', water_levels_of_interest)
    
    # Define user-specified y-axis limits
    primary_y_min, primary_y_max = -50, 0  # Limits for the primary y-axis
    secondary_y_min, secondary_y_max = -500, 22500  # Limits for the secondary y-axis

    # Dynamically build metrics list based on water levels
    percent_metrics = [f"% Above {wl}" for wl in water_levels_of_interest]
    export_metrics = ['Dry Exports', 'Wet Exports']
    metrics = percent_metrics + export_metrics

    # Example usage
    policies_to_plot = get_policies_to_plot()
    pt_policies = get_pt_policies()
    unique_colors = get_unique_colors(len(policies_to_plot))

    # Determine the number of rows and columns
    num_rows = len(pt_policies)  # One row per post-transition policy
    num_cols = len(ssps_to_plot)  # One column per SSP

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(23, 6 * num_rows), sharey=False)

    # Iterate through post-transition policies (rows)
    for row, pt_policy in enumerate(pt_policies):
        # Iterate through SSPs (columns)
        for col, ssp in enumerate(ssps_to_plot):
            ax = axes[row, col] if num_rows > 1 else axes[col]  # Handle single row layout
            ax2 = ax.twinx()  # Add secondary y-axis for export metrics
            lines = []
            labels = []

            #### no export no post-transition
            # Select data for the specific combination and year slice
            tmp_array = Policy_Outputs.sel(ssp=ssp, PT_policy='None', policy='A2').isel(
                year=slice(start_year_index, end_year_index)
            )

            # Calculate metrics
            no_exports_no_pt_percent = {}

            # Percentage of GCMs above user-defined water levels
            for wl in water_levels_of_interest:
                no_exports_no_pt_percent[f"% Above {wl}"] = (
                        (tmp_array['Water_Level'] >= wl).sum(dim='GCM') / len(tmp_array['GCM']) * 100
                    ).mean(dim='year').values



            for tmp_policy in policies_to_plot:
                try:
                    # Select data for the specific combination and year slice
                    tmp_array = Policy_Outputs.sel(ssp=ssp, PT_policy=pt_policy, policy=tmp_policy).isel(
                        year=slice(start_year_index, end_year_index)
                    )

                    tmp_array_all_years = Policy_Outputs.sel(ssp=ssp, PT_policy=pt_policy, policy=tmp_policy).isel(
                        year=slice(0, 77)
                    )

                    # Calculate metrics
                    metrics_data = {}

                    # Calculate the percentage of GCMs above the water level of interest
                    for wl in water_levels_of_interest:

                        tmp_val_for_policy = (
                            (tmp_array['Water_Level'] >= wl).sum(dim='GCM') / len(tmp_array['GCM']) * 100
                        ).mean(dim='year').values

                        ### get difference from no exports no post-transition
                        diff = tmp_val_for_policy - no_exports_no_pt_percent[f"% Above {wl}"]

                        metrics_data[f"% Above {wl}"] = diff

                    # Dry and wet year exports
                    year_types = ryt_sef_xarray.sel(ssp=ssp).isel(year=slice(start_year_index, end_year_index))['Year_Type']
                    dry_mask = np.isin(year_types, dry_types)
                    wet_mask = np.isin(year_types, wet_types)

                    metrics_data['Dry Exports'] = tmp_array['Exports'].where(dry_mask).mean(dim=['GCM', 'year'], skipna=True).values
                    metrics_data['Wet Exports'] = tmp_array['Exports'].where(wet_mask).mean(dim=['GCM', 'year'], skipna=True).values

                    # Add to lines
                    lines.append(metrics_data)
                    labels.append(f"{tmp_policy}/{pt_policy}")

                except Exception as e:
                    # Skip if no data or an error occurs
                    print(f"Skipping {tmp_policy}/{pt_policy} for {ssp}: {e}")
                    continue

            # Plot lines
            for idx, metrics_data in enumerate(lines):
                # Split metrics for the two y-axes
                percent_values = [metrics_data[m] for m in percent_metrics]
                export_values = [metrics_data[m] for m in export_metrics]

                # Apply scaling to secondary y-axis (invert values for plotting)
                export_values = np.array([(elem * -1) for elem in export_values])

                # Define x positions for percent and export metrics
                x_percent = range(len(percent_metrics))  # Positions for percent metrics
                x_export = range(len(percent_metrics), len(percent_metrics) + len(export_metrics))  # Positions for export metrics

                # Plot percent metrics on the left y-axis
                ax.plot(x_percent, percent_values, linewidth=2, color=unique_colors[idx], alpha=1.0)

                # Plot export metrics on the right y-axis
                ax2.plot(x_export, export_values, color=unique_colors[idx], linewidth=2, alpha=1.0)

                # Add a transition line connecting the last point on the left axis to the first point on the right axis
                transition_x = [x_percent[-1], x_export[0]]  # X-coordinates for the transition

                # Map the secondary y-axis value to the primary y-axis scale using fixed limits
                adjusted_right_y = primary_y_min + ((export_values[0] - (-secondary_y_max)) / (secondary_y_max - secondary_y_min)) * (primary_y_max - primary_y_min)

                # Define the transition y-values
                transition_y = [percent_values[-1], adjusted_right_y]

                tmp_policy_label = policies_to_plot[idx]
                # Plot the transition line
                ax.plot(transition_x, transition_y, linestyle='-', color=unique_colors[idx], linewidth=2, alpha=1.0,label=tmp_policy_label)

            if (row == 0) and (col == 1):
                ax.legend(bbox_to_anchor=(0.0, 1.8), loc='upper left', ncol=2, fontsize=22)

            # Set fixed y-axis limits
            ax.set_ylim(primary_y_min, primary_y_max)
            ax2.set_ylim(-secondary_y_max, -secondary_y_min)  # Keep secondary axis flipped

            # Add unified x-ticks
            combined_metrics = percent_metrics + export_metrics
            ax.set_xticks(range(len(combined_metrics)))
            ax.set_xticklabels(combined_metrics, rotation=45, ha='right')

            # Add titles, labels, and grid
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.grid(axis='x', linestyle='-', alpha=1.0)

            if col == 0:
                ax.set_ylabel(f"PT Policy\n{pt_policy}", fontsize=24, fontweight='bold',labelpad=40, va='center')
            if row == 0:
                ax.set_title(f"{ssp}", fontsize=24,fontweight='bold')

            # Set custom tick labels for the secondary y-axis
            yticks = np.linspace(-secondary_y_max, 0, 10)  # Generate 10 evenly spaced ticks
            ax2.set_yticks(yticks)
            ax2.set_yticklabels([f"{abs(int(tick))}" for tick in yticks])  # Apply absolute value to tick labels

            #ax.tick_params(axis='both', labelsize=18)  # Increase xtick label size
            #ax.tick_params(axis='y', labelsize=27)  # Increase ytick label size


            # Add explicit y-ticks to include the first and last values
            primary_yticks = ax.get_yticks()
            if primary_yticks[0] != primary_y_min:
                primary_yticks = np.insert(primary_yticks, 0, primary_y_min)  # Add the min value if missing
            if primary_yticks[-1] != primary_y_max:
                primary_yticks = np.append(primary_yticks, primary_y_max)  # Add the max value if missing
            ax.set_yticks(primary_yticks)
            ax.set_yticklabels([f"{int(tick)}" for tick in primary_yticks], fontsize=16)  # Customize label size

            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.tick_params(axis='both', which='minor', labelsize=18)
            ax2.tick_params(axis='y', labelsize=18)  # Increase ytick label size

    # Add a global title and layout adjustment
    start_yr_for_analysis = Start_Year + start_year_index
    end_yr_for_analysis = Start_Year + end_year_index
    fig.text(-0.06,0.35,'Percent Difference in GCMs Above\nWater Level Compared to No Exports (A2+None) (%)',fontsize=28,rotation=90,fontweight='bold',va='center',ha='center')
    fig.text(1.04,0.1,'Average Exports/Year (ac-ft)',fontsize=28,rotation=270,fontweight='bold')
    fig.suptitle(f"Performance of Policies for Selected Metrics Relative to No Exports in Pre- and Post-Transition\nEvaluated for {start_yr_for_analysis}-{end_yr_for_analysis} ({start_year_index}-{end_year_index} Years From Now)", y=0.91,fontsize=28,fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

### Performance for Selected Policies in a table format
def table_plot_policy_performance(
    Policy_Outputs, 
    start_year_index, 
    end_year_index, 
    get_policies_to_plot, 
    ssps_to_plot, 
    get_pt_policies, 
    water_levels_of_interest_str, 
    Start_Year,
    ryt_sef_xarray):
    
    water_levels_of_interest = [int(item) for item in water_levels_of_interest_str.split()]
    print('Water levels of interest:', water_levels_of_interest)
    
    # Dynamically build metrics list based on water levels
    percent_metrics = [f"% Above {wl}" for wl in water_levels_of_interest]
    percent_metrics += [f'% GCMs Transition\nduring 0 to {end_year_index}-Yrs']
    export_metrics = ['Mean Exports','Dry Exports', 'Wet Exports']

    # Initialize lists to store extracted metrics data
    percent_data_records = []
    export_data_records = []

    ### does transition occur?
    transition_time_threshold = end_year_index

    # Initialize lists to store extracted metrics data
    percent_data_records = []
    export_data_records = []
    
    policies_to_plot = get_policies_to_plot()
    pt_policies = get_pt_policies()

    # Extract data without taking differences
    for pt_policy in pt_policies:
        for ssp in ssps_to_plot:
            for pre_policy in policies_to_plot:
                try:
                    # Select data for the specific combination and year slice
                    tmp_array = Policy_Outputs.sel(ssp=ssp, PT_policy=pt_policy, policy=pre_policy).isel(
                        year=slice(start_year_index, end_year_index)
                    )

                    tmp_array_all_years = Policy_Outputs.sel(ssp=ssp, PT_policy=pt_policy, policy=pre_policy).isel(
                        year=slice(0, end_year_index)
                    )

                    # Calculate metrics data
                    metrics_data = {}

                    # Calculate percentage of GCMs above user-defined water levels
                    for wl in water_levels_of_interest:
                        metrics_data[f"% Above {wl}"] = float((
                            (tmp_array['Water_Level'] >= wl).sum(dim='GCM') / len(tmp_array['GCM']) * 100).mean(dim='year').values)

                    # Calculate GCM transition percentage
                    pct_gcm_transition = (
                        (tmp_array_all_years['Transition_Time'] < transition_time_threshold).sum() /
                        len(tmp_array_all_years['GCM'])
                    ).values * 100
                    metrics_data[f'% GCMs Transition\nduring 0 to {end_year_index}-Yrs'] = pct_gcm_transition

                    # # Dry and wet year exports based on all years up to `end_year_index`
                    # year_types = ryt_sef_xarray.sel(ssp=ssp).isel(year=slice(0, end_year_index))['Year_Type']
                    # dry_mask = np.isin(year_types, dry_types)
                    # wet_mask = np.isin(year_types, wet_types)

                    # metrics_data['Mean Exports'] = float(tmp_array_all_years['Exports'].mean(dim=['GCM', 'year'], skipna=True).values)
                    # metrics_data['Dry Exports'] = float(tmp_array_all_years['Exports'].where(dry_mask).mean(dim=['GCM', 'year'], skipna=True).values)
                    # metrics_data['Wet Exports'] = float(tmp_array_all_years['Exports'].where(wet_mask).mean(dim=['GCM', 'year'], skipna=True).values)

                    # Dry and wet year exports based on all years up to `end_year_index`
                    year_types = ryt_sef_xarray.sel(ssp=ssp).isel(year=slice(start_year_index, end_year_index))['Year_Type']
                    dry_mask = np.isin(year_types, dry_types)
                    wet_mask = np.isin(year_types, wet_types)

                    metrics_data['Mean Exports'] = float(tmp_array['Exports'].mean(dim=['GCM', 'year'], skipna=True).values)
                    metrics_data['Dry Exports'] = float(tmp_array['Exports'].where(dry_mask).mean(dim=['GCM', 'year'], skipna=True).values)
                    metrics_data['Wet Exports'] = float(tmp_array['Exports'].where(wet_mask).mean(dim=['GCM', 'year'], skipna=True).values)

                    # Append results to records
                    policy_label = f"{pre_policy}-{pt_policy}"
                    percent_data_records.append([policy_label] + [metrics_data[m] for m in percent_metrics])
                    export_data_records.append([policy_label] + [metrics_data[m] for m in export_metrics])

                except Exception as e:
                    print(f"Skipping {pre_policy}-{pt_policy} for {ssp}: {e}")
                    continue

    # Convert to DataFrames
    percent_data_df = pd.DataFrame(percent_data_records, columns=['Policy Combination'] + percent_metrics).set_index('Policy Combination')
    export_data_df = pd.DataFrame(export_data_records, columns=['Policy Combination'] + export_metrics).set_index('Policy Combination')

    # Sort by "% Above 6391"
    sorted_percent_data_df = percent_data_df.sort_values(by="% Above 6391", ascending=False)
    sorted_export_data_df = export_data_df.loc[sorted_percent_data_df.index]

    # Plot heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, len(pt_policies)*5), gridspec_kw={'width_ratios': [len(percent_metrics), len(export_metrics)]})

    # Percent metrics heatmap
    sns.heatmap(
        sorted_percent_data_df, 
        annot=True, 
        cmap='Blues', 
        vmin=10, 
        vmax=80, 
        linewidths=0.5, 
        ax=ax1, 
        cbar=False,
        annot_kws={"size": 14}  # Increase font size
    )

    # Ensure y-tick locations match DataFrame length
    ytick_locs = np.arange(len(sorted_percent_data_df)) + 0.5  # Add 0.5 to center the labels
    ax1.set_yticks(ytick_locs)
    ax1.set_yticklabels(sorted_percent_data_df.index, fontsize=16)

    # Export metrics heatmap
    sns.heatmap(
        sorted_export_data_df, 
        annot=True, 
        cmap='Reds', 
        vmin=0, 
        vmax=20000, 
        linewidths=0.5, 
        ax=ax2, 
        cbar=False, 
        yticklabels=False,
        fmt=",.0f",  # Disable scientific notation
        annot_kws={"size": 14}  # Increase font size
    )

    # Ensure y-tick locations match DataFrame length
    ytick_locs = np.arange(len(sorted_export_data_df)) + 0.5  # Add 0.5 to center the labels
    ax2.set_yticks(ytick_locs)
    ax2.set_yticklabels(sorted_export_data_df.index, fontsize=16)

    # Add colorbars
    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=10, vmax=80), cmap='Blues'), 
        cax=cbar_ax1, label="Percent of GCMs (%)"
    ).ax.set_ylabel("Percent of GCMs", fontsize=16, fontweight='bold', labelpad=15)

    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.02, 0.35])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=20000), cmap='Reds'), 
        cax=cbar_ax2, label="Export Volume (ac-ft)"
    ).ax.set_ylabel("Export Volume (ac-ft)", fontsize=16, fontweight='bold', labelpad=15)

    # Customize labels
    ax1.set_title(f"Percent Metrics (based on {start_year_index} to {end_year_index} Years)", fontsize=18, fontweight='bold')
    ax2.set_title(f"Export Metrics (based on {start_year_index} to {end_year_index} Years)", fontsize=18, fontweight='bold')

    ax1.set_ylabel("Policy Combinations (Pre-Post Transition)", fontsize=22, fontweight='bold')
    ax2.set_ylabel("")

    ax1.set_xticks(np.arange(len(percent_metrics)) + 0.5)
    ax1.set_xticklabels(percent_metrics, rotation=45, ha='right', fontsize=18)
    ax2.set_xticks(np.arange(len(export_metrics)) + 0.5)
    ax2.set_xticklabels(export_metrics, rotation=45, ha='right', fontsize=18)

    ax1.set_yticklabels(sorted_percent_data_df.index, fontsize=16)

    start_yr_for_analysis = Start_Year + start_year_index
    end_yr_for_analysis = Start_Year + end_year_index
    fig.suptitle(
        f"Performance of Policies for {ssps_to_plot[0]}\nEvaluated for {start_yr_for_analysis}-{end_yr_for_analysis} ({start_year_index}-{end_year_index} Years From Now)", 
        y=1.0, fontsize=28, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 0.9, 1])