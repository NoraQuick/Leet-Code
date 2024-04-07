#Day 1: Add Two Integers
#Runtime: 33ms
class Solution:
    def sum(self, num1: int, num2: int) -> int:
        return num1 + num2
    
#Day 2: Root Equals Sun of Children
#Runtime: 38ms
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def checkTree(self, root: Optional[TreeNode]) -> bool:
        return root.val == root.left.val + root.right.val
    
#Day 3: Running Sum of 1d Array
#Runtime 19ms
class Solution(object):
    def runningSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for i in range(1, len(nums)):
            nums[i] += nums[i-1]
        return nums
    
#Day 4: Richest Customer Wealth
#Runtime: 31ms
class Solution(object):
    def maximumWealth(self, accounts):
        """
        :type accounts: List[List[int]]
        :rtype: int
        """
        return max([sum(i) for i in accounts])
    
#Day 5: Fizz Buzz
#Runtime: 22ms
class Solution(object):
    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        ans = []

        for i in range(1, n+1):
            if i%3 == 0 and i%5 == 0:
                ans.append("FizzBuzz")
            elif i%3 == 0:
                ans.append("Fizz")
            elif i%5 == 0:
                ans.append("Buzz")
            else:
                ans.append(str(i))
        
        return ans
    
#Day 6: Number of Steps to Reduce a Number to Zero
#Runtime: 4ms
class Solution(object):
    def numberOfSteps(self, num):
        """
        :type num: int
        :rtype: int
        """
        remain = 0
        while num > 0:
            if num%2 != 0:
                num -= 1
            else:
                num /= 2
            remain += 1
        return remain
    
#Day 7: Ransom Note
#Runtime: 66ms
class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        r, m = Counter(ransomNote), Counter(magazine)

        if r & m == r:
            return True
        return False
    
#Day 8: Range Sum of BST
#Runtime: 110ms
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        if not root or low > high:
            return 0
        
        if root.val < low:
            return self.rangeSumBST(root.right, low, high)
        elif root.val > high:
            return self.rangeSumBST(root.left, low, high)
        else:
            return root.val + self.rangeSumBST(root.left, low, high) + self.rangeSumBST(root.right, low, high)
    
#Day 9: Leaf-Sumilar Trees
#Runtime: 37ms
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def dfs(root):
            if root is None:
                return []

            leaves = dfs(root.left) + dfs(root.right)

            return leaves or [root.val]

        if dfs(root1) == dfs(root2):
            return True
        else:
            return False
    
#Day 10: Amount of Time for Binary Tree to Be Infected
#Runtime: 407ms
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
        def dfs(node):
            if node is None:
                return
            if node.left:
                graph[node.val].append(node.left.val)
                graph[node.left.val].append(node.val)
            if node.right:
                graph[node.val].append(node.right.val)
                graph[node.right.val].append(node.val)

            dfs(node.left)
            dfs(node.right)

        graph = defaultdict(list)

        dfs(root)

        infected = set()
        queue = deque([start])
        time = -1

        while queue:
            time += 1
            for _ in range(len(queue)):
                current_node = queue.popleft()
                infected.add(current_node)
                for neighbor in graph[current_node]:
                    if neighbor not in infected:
                        queue.append(neighbor)

        return time
    
#Day 11: Leaf-Similar Trees
#Runtime: 42ms
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def dfs(root):
            if root is None:
                return []

            leaves = dfs(root.left) + dfs(root.right)

            return leaves or [root.val]

        if dfs(root1) == dfs(root2):
            return True
        else:
            return Falsex
    
#Day 12: Maximum Difference Between Node and Ancestor
#Runtime: 47ms
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def dfs(self, root, min_val, max_val):
            if not root:
                return 0
            
            min_val = min(root.val, min_val)
            max_val = max(root.val, max_val)

            if not root.left and not root.right:
                return max_val - min_val

            return max(self.dfs(root.left, min_val, max_val), self.dfs(root.right, min_val, max_val))
            
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        return self.dfs(root, root.val, root.val)
    
#Day 13: Determine if String Halves Are Alike
#Runtime: 30ms
class Solution:
    def halvesAreAlike(self, s: str) -> bool:
        def checkVow(s, vowels):
            final = [each for each in s if each in vowels]
            return len(final)
        
        vowels = "aeiou"

        str_a = s[0:len(s)//2].lower()
        str_b = s[len(s)//2:].lower()

        a_num_vow = checkVow(str_a, vowels)
        b_num_vow = checkVow(str_b, vowels)

        if a_num_vow == b_num_vow:
            return True
        else:
            return False
    
#Day 14: Insert Delete GetRandom O(1)
#Runtime: 265ms
import random

class RandomizedSet:

    def __init__(self):
        self.lst = []
        self.idx_map = {}

    def search(self, val):
        return val in self.idx_map

    def insert(self, val: int) -> bool:
        if self.search(val):
            return False

        self.lst.append(val)
        self.idx_map[val] = len(self.lst) - 1

        return True


    def remove(self, val: int) -> bool:
        if not self.search(val):
            return False

        idx = self.idx_map[val]

        self.lst[idx] = self.lst[-1]
        self.idx_map[self.lst[-1]] = idx
        self.lst.pop()

        del self.idx_map[val]

        return True
        

    def getRandom(self) -> int:
        return random.choice(self.lst)
    
#Day 15: First Unique Character in a String
#Runtime: 96ms
class Solution:
    def firstUniqChar(self, s: str) -> int:
        mp = {}

        for x in s:
            mp[x] = mp.get(x, 0) + 1

        for i in range(len(s)):
            if mp[s[i]] == 1:
                return i

        return -1
    
#Day 16: Group Anagrams
#Runtime: 77ms
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups = {}

        for s in strs:
            key = "".join(sorted(s))

            if key not in groups:
                groups[key] = [s]
            else:
                groups[key].append(s)

        return groups.values()
    
#Day 17: Sort Characters By Frequency
#Runtime: 44ms
from collections import Counter, OrderedDict

class Solution:
    def frequencySort(self, s: str) -> str:
        mp = Counter(s)

        y = OrderedDict(sorted(mp.items(), key=lambda x: x[1], reverse=True))

        des_s = ''.join([char * freq for char, freq in y.items()])

        return des_s
    
#Day 18: Perfect Squares
#Runtime: 2196ms
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [0] * (n+1)
        dp[0] = 0
        dp[1] = 1

        for i in range(2, n+1):
            min_val = float('inf')

            for j in range(1, int(i ** 0.5) + 1):
                rem = i - j * j
                min_val = min(min_val, dp[rem])

            dp[i] = min_val + 1

        return dp[n]
    
#Day 19: Largest Divisible Subset
#Runtime: 222ms
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums.sort()
        n = len(nums)
        dp = [1] * n
        max_size = 1
        max_index = 0

        for i in range(1, n):
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    dp[i] = max(dp[i], dp[j] + 1)
                    if dp[i] > max_size:
                        max_size = dp[i]
                        max_index = i
        
        result = []
        num = nums[max_index]

        for i in range(max_index, -1, -1):
            if num % nums[i] == 0 and dp[i] == max_size:
                result.append(nums[i])
                num = nums[i]
                max_size -= 1

        return result
    
#Day 20: Majority Element
#Runtime: 131ms
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        ret = 0
        freq = 0

        for n in nums:
            if freq == 0:
                ret = n

            if n == ret:
                freq += 1
            else:
                freq -= 1

        return ret

#Day 21: Find First Palindromic String in the Array
#Runtime: 67ms
class Solution:
    def firstPalindrome(self, words: List[str]) -> str:
        for word in words:
            if word == word[::-1]:
                return word
            
        return ""

#Day 22: Rearrange Array Elements by Sign
#Runtime: 997ms
class Solution:
    def rearrangeArray(self, nums: List[int]) -> List[int]:
        ans = [0] * len(nums)
        pos = 0
        neg = 1

        for num in nums:
            if num > 0:
                ans[pos] = num
                pos += 2
            else:
                ans[neg] = num
                neg += 2

        return ans

#Day 23: Find Polygon With the Largest Perimeter
#Runtime: 504ms
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()

        s = sum(nums)
        n = len(nums)

        for i in range(n - 1, 1, -1):
            s -= nums[i]
            if s > nums[i]:
                return s + nums[i]

        return -1

#Day 24: Least Number of Unique Integers after K Removals
#Runtime: 330ms
class Solution:
    def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
        mp = collections.Counter(arr)
        v = list(mp.values())
        c = 0
        v.sort()

        for i in range(len(v)):
            if k > v[i]:
                k -= v[i]
                v[i] = 0

            else:
                v[i] -= k
                k = 0

            if v[i] != 0:
                c += 1

        return c

#Day 25: Missing Number
#Runtime: 122ms
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        i = 0

        if nums[-1] != n:
            return n

        for i in range(0, len(nums)):
            if nums[i] != i:
                return i

        return 0

#Day 26: Bitwise AND of Numbers Range
#Runtime: 47ms
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        c = 0

        while left != right:
            left >>= 1
            right >>= 1
            c += 1

        return left << c

#Day 27: Find the Town Judge
#Runtime: 620ms
class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        trusted = [0] * (n + 1)

        for a, b in trust:
            trusted[a] -= 1
            trusted[b] += 1

        for i in range(1, n + 1):
            if trusted[i] == n - 1:
                return i

        return -1

#Day 28: Cheapest Flights Within K Stops
#Runtime: 95ms
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        adj = [[] for _ in range(n)]

        for flight in flights:
            adj[flight[0]].append((flight[1], flight[2]))

        q = [(src, 0)]
        minCost = [float('inf') for _ in range(n)]
        stops = 0

        while q and stops <= k:
            size = len(q)

            for i in range(size):
                currNode, cost = q.pop(0)

                for neighbor, price in adj[currNode]:
                    if price + cost >= minCost[neighbor]:
                        continue
                    minCost[neighbor] = price + cost
                    q.append((neighbor, minCost[neighbor]))

            stops += 1

        return -1 if minCost[dst] == float('inf') else minCost[dst]

#Day 29: Same Tree
#Runtime: 38ms
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True

        if p and q and p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

        return False

#Day 30: Diameter of Binary Tree
#Runtime: 44ms
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        def diameterCalc(node, res):
            if not node:
                return 0

            left = diameterCalc(node.left, res)
            right = diameterCalc(node.right, res)

            res[0] = max(res[0], left + right)

            return max(left, right) + 1

        res = [0]

        diameterCalc(root, res)

        return res[0]

#Day 31: Find Bottom Left Tree Valuue
#Runtime: 40ms
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        queue = deque([root])
        leftmost = None

        while queue:
            node = queue.popleft()

            leftmost = node.val

            if node.right:
                queue.append(node.right)
            if node.left:
                queue.append(node.left)

        return leftmost

#Day 32: Even Odd Tree
#Runtime: 201ms
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def isEvenOddTree(self, root: Optional[TreeNode]) -> bool:
        queue = deque([root])
        level = 0

        if not root:
            return True

        while queue:
            prev_val = None

            for _ in range(len(queue)):
                node = queue.popleft()

                if (level % 2 == 0 and (node.val % 2 == 0 or (prev_val is not None and node.val <= prev_val))) or \
                (level % 2 == 1 and (node.val % 2 == 1 or (prev_val is not None and node.val >= prev_val))):
                    return False

                prev_val = node.val

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            level += 1

        return True

#Day 33: Maximum Odd Binary Number
#Runtime: 40ms
class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        sort_s = sorted(s, reverse=True)

        for i in range(len(s) -1, -1, -1):
            if sort_s[i] == '1':
                sort_s[i], sort_s[-1] = sort_s[-1], sort_s[i]
                break

        return ''.join(sort_s)

#Day 34: Bag of Tokens
#Runtime: 51ms
class Solution:
    def bagOfTokensScore(self, tokens: List[int], power: int) -> int:
        tokens.sort()

        n = len(tokens)
        score = 0
        max_score = 0
        face_up = 0
        face_down = n - 1

        while face_up <= face_down:
            if power >= tokens[face_up]:
                power -= tokens[face_up]
                score += 1
                face_up += 1
                max_score = max(max_score, score)
            elif score > 0:
                power += tokens[face_down]
                score -= 1
                face_down -= 1
            else:
                break

        return max_score

#Day 35: Linked List Cycle
#Runtime: 45ms
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        x = head
        y = head

        if not head or not head.next:
            return False

        while y.next and y.next.next:
            y = y.next.next
            x = x.next

            if x == y:
                return True

        return False
    
#Day 36: Middle of the Linked List
#Runtime: 34ms
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast = head
        slow = head

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        return slow

#Day 37: Count Elements With Maximum Frequency
#Runtime: 51ms
class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        c = Counter(nums)
        max_freq = max(c.values())

        return sum(freq for freq in c.values() if freq == max_freq)

#Day 38: Custom Sort String
#Runtime: 26ms
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        result = ""
        mp = {}

        for char in s:
            mp[char] = mp.get(char, 0) + 1
        for char in order:
            if char in mp:
                result += char * mp[char]
                del mp[char]
        for char, count in mp.items():
            result += char * count

        return result

#Day 39: Remove Zero Sum Consecutive Nodes from Linked List
#Runtime: 43ms
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeZeroSumSublists(self, head: Optional[ListNode]) -> Optional[ListNode]:
        temp = ListNode(0)
        temp.next = head
        total_sum = 0
        sums = {0: temp}
        current = head

        while current:
            total_sum += current.val
            if total_sum in sums:
                delete = sums[total_sum].next
                temp_sum = total_sum + delete.val
                while delete != current:
                    del sums[temp_sum]
                    delete = delete.next
                    temp_sum += delete.val
                
                sums[total_sum].next = current.next
            else:
                sums[total_sum] = current

            current = current.next

        return temp.next
    
#Day 40: Find the Pivot Integer
#Runtime: 38ms
class Solution:
    def pivotInteger(self, n: int) -> int:
        x = sqrt(n * (n + 1) / 2)

        if x % 1 != 0:
            return -1
        else:
            return int(x)

#Day 41: Binary Subarrays With Sum
#Runtime: 232ms
class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        count = {0: 1}
        curr_sum = 0
        total_sum = 0

        for num in nums:
            curr_sum += num
            if curr_sum - goal in count:
                total_sum += count[curr_sum - goal]

            count[curr_sum] = count.get(curr_sum, 0) + 1

        return total_sum

#Day 42: Product of Array Except Self
#Runtime: 175ms
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        pre = [1] * n
        suf = [1] * n

        for i in range(1, n):
            pre[i] = pre[i - 1] * nums[i - 1]

        for i in range(n - 2, -1, -1):
            suf[i] = suf[i + 1] * nums[i + 1]

        answer = [pre[i] * suf[i] for i in range(n)]

        return answer

#Day 43: Minimum Number of Arrows to Burst Balloons
#Runtime: 1034ms
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda x: x[0])
        arrow = 1
        end = points[0][1]

        for balloon in points[1:]:
            if balloon[0] > end:
                arrow += 1
                end = balloon[1]
            else:
                end = min(end, balloon[1])

        return arrow

#Day 44: Task Scheduler
#Runtime: 356ms
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        freq = [0] * 26

        for task in tasks:
            freq[ord(task) - ord('A')] += 1

        freq.sort()
        chunk = freq[25] - 1
        idle = chunk * n

        for i in range(24, -1, -1):
            idle -= min(chunk, freq[i])

        return len(tasks) + idle if idle >= 0 else len(tasks)

#Day 45: Find All Duplicates in an Array
#Runtime: 266ms
class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        two_int = []
        n = len(nums)

        for x in nums:
            x = abs(x)
            if nums[x - 1] < 0:
                two_int.append(x)

            nums[x-1] *= -1

        return two_int
    
#Day 46: First Missing Positive
#Runtime: 318ms
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)

        def swap(arr, i, j):
            arr[i], arr[j] = arr[j], arr[i]

        for i in range(n):
            while 0 < nums[i] <= n and nums[i] != nums[nums[i] - 1]:
                swap(nums, i, nums[i]- 1)

        for i in range(n):
            if nums[i] != i + 1:
                return i + 1

        return n + 1

#Day 47: Subarray Product Less Than K
#Runtime: 489ms
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        left, right = 0, 0
        product = 1
        count = 0
        n = len(nums)

        if k <= 1:
            return 0

        while right < n:
            product *= nums[right]
            while product >= k:
                product //= nums[left]
                left += 1
            
            count += 1 + (right - left)
            right += 1

        return count
    
#Day 48: Length of Longest Subarray With at Most K Frequency
#Runtime: 1159ms
class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        ans = 0
        mp = {}
        l = 0
        n = len(nums)

        for r in range(n):
            mp[nums[r]] = mp.get(nums[r], 0) + 1

            if mp[nums[r]] > k:
                while nums[l] != nums[r]:
                    mp[nums[l]] -= 1
                    l += 1
                mp[nums[l]] -= 1
                l += 1
            ans = max(ans, r - l + 1)
        return ans

#Day 49: Count Subarray Where Max Element Appears at Least K Times
#Runtime: 876ms
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans, l, r = 0, 0, 0
        mx = max(nums)
        n = len(nums)

        while r < n:
            k -= nums[r] == mx
            r += 1
            while k == 0:
                k += nums[l] == mx
                l += 1

            ans += l

        return ans
    
#Day 50: Length of Last Word
#Runtime: 36ms
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        words = s.strip().split()

        if not words:
            return 0

        return len(words[-1])    

#Day 51: Word Search
#Runtime: 3248ms
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(board, word, i, j, idx):
            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[idx]:
                return False

            if idx == len(word) - 1:
                return True

            tmp = board[i][j]
            board[i][j] = "*"

            res = dfs(board, word, i+1, j, idx+1) or dfs(board, word, i-1, j, idx+1) \
                or dfs(board, word, i, j+1, idx+1) or dfs(board, word, i, j-1, idx+1)

            board[i][j] = tmp

            return res

        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(board, word, i, j, 0):
                    return True

        return False
    
#Day 52: Maximum Nesting Depth of the Parentheses
#Runtime: 43ms
class Solution:
    def maxDepth(self, s: str) -> int:
        count = 0
        max_num = 0

        for i in s:
            if i == "(":
                count += 1
                
                if max_num < count:
                    max_num = count

            if i == ")":
                count -= 1

        return max_num
