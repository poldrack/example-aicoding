def merge_sort(lst):

    if len(lst) <= 1:      # Base case: a single or empty list is considered sorted

        return lst



    mid = len(lst) // 2    # Calculate the middle index for splitting the list

    left = lst[:mid]       # Split the list into two halves

    right = lst[mid:]



    left = merge_sort(left)     # Recursively sort the left half

    right = merge_sort(right)   # Recursively sort the right half



    return merge(left, right)   # Merge the two sorted halves





def merge(left, right):

    merged = []

    left_index = 0

    right_index = 0



    # Continue merging until either one of the halves is exhausted

    while left_index < len(left) and right_index < len(right):

        if left[left_index] < right[right_index]:

            merged.append(left[left_index])

            left_index += 1

        else:

            merged.append(right[right_index])

            right_index += 1



    # Add the remaining elements from the left half

    while left_index < len(left):

        merged.append(left[left_index])

        left_index += 1



    # Add the remaining elements from the right half

    while right_index < len(right):

        merged.append(right[right_index])

        right_index += 1



    return merged
