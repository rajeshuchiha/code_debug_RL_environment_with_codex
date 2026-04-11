def longest_unique_substring(text):
    seen = set()
    best = 0
    left = 0
    for right, char in enumerate(text):
        if char in seen:
            left += 1
        seen.add(char)
        best = max(best, right - left + 1)
    return best
