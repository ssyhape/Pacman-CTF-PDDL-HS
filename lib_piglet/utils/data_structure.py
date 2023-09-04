# utils.binary_heap/Binary_Heap.py
# Implements a binary heap data structure. It maintain the data in a priority order and pop the top (min) item.
#

#
# @author: mike
# @created: 2020-07-16
#
from collections import deque
from typing import Callable


# A stack push new item to the end of the stack and pop item from the end of the stack.
class stack(deque):

    # Push item to the end of the stack.
    def push(self, item):
        self.append(item)


# A queue push new item to the start of the queue and pop item from the end of the queue.
class queue(deque):

    # Push item to the start of the stack.
    def push(self, item):
        self.appendleft(item)

class heap_item:

    def __init__(self, item, handle_id):
        self.item_: object = item
        self.handle_id_: int = handle_id

    def __eq__(self, other):
        return self.item_ == other.item_

    def __repr__(self):
        return self.item_.__repr__()

    def __str__(self):
        return self.item_.__str__()


class bin_heap:
    """
    Binary heap maintain highest priority items at the front of the queue
    and the lowest priority items at the back.
    """

    def __init__(self, compare_function):
        """
        Initiate binary heap
        :param compare_function: A function that return true if give two search_nodes as arguments and node1 >= node2.
        """

        self.heapList: list  = [0]
        self.currentSize: int  = 0
        self.compare_function: Callable = compare_function

        self.current_id: int = 0
        self.handle: dict ={}


    def insert(self, item):
        """
        Insert an item to the heap, and maintain heap structure.
        :param item:
        :return handle_id: The handle of insert item.
        """
        id = self.current_id
        self.heapList.append(heap_item(item,id))
        self.currentSize = self.currentSize + 1
        self.handle[id] = self.currentSize
        self.current_id += 1
        self.__percUp(self.currentSize)
        return id

    def push(self, item):
        """
        Do same thing as insert. Push an item into the heap and update the structure
        :param item:
        :return handle_id: return the handle to heap item
        """
        return self.insert(item)

    def pop(self):
        """
        Pop the top item of the heap and update the heap to maintain heap strucure.
        :return:
        """
        retval: heap_item = self.heapList[1]
        self.handle.pop(retval.handle_id_)
        self.heapList[1] = self.heapList[self.currentSize]
        self.handle[self.heapList[1].handle_id_] = 1
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.__percDown(1)
        return retval.item_

    def build(self, alist:list):
        """
        Use a list to initialize the heap.
        :param alist:
        :return list: A list of handle_id
        """
        if type(alist) != list:
            raise Exception("Given argument is not list")
        self.current_id = 0
        self.handle.clear()
        i = len(alist) // 2
        self.heapList = [0] + [heap_item(x,0) for x in alist]
        self.currentSize = len(alist)
        for i in range(1,self.currentSize+1):
            self.handle[self.current_id] = self.heapList[i]
            self.heapList[i].handle_id_ = self.current_id
            self.current_id += 1
        while (i > 0):
            self.__percDown(i)
            i = i - 1
        return list(range(0,self.currentSize+1))

    def clear(self):
        """
        Clear the heap.
        :return:
        """
        self.heapList = [0]
        self.currentSize = 0
        self.current_id = 0
        self.handle.clear()

    def erase(self, handle_id: int):
        if not handle_id in self.handle:
            raise KeyError("Handle id does not exist")
        index = self.handle[handle_id]
        retval: heap_item = self.heapList[index]
        self.handle.pop(retval.handle_id_)
        self.heapList[index] = self.heapList[self.currentSize]
        self.handle[self.heapList[index].handle_id_] = index
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.__percDown(index)
        return retval.item_

    def increase(self,handle_id: int):
        """
        Update the item, if the target's value increased.
        :param handle_id:
        :return:
        """
        if handle_id in self.handle:
            self.__percDown(self.handle[handle_id])
        else:
            raise ValueError("Given item not in the heap")

    def decrease(self,handle_id: int):
        """
        Update the item, if the target's value decreased.
        :param handle_id:
        :return:
        """
        if handle_id in self.handle:
            self.__percUp(self.handle[handle_id])
        else:
            raise ValueError("Given item not in the heap")

    def update(self):
        i = self.currentSize // 2
        while (i > 0):
            self.__percDown(i)
            i = i - 1

    def size(self):
        """
        Return the size of the heap.
        :return int: size of the heap
        """
        return self.currentSize

    def empty(self):
        """
        Return true if heap is empty
        :return bool: is the heap empty.
        """
        return len(self.heapList) == 1

    def __percUp(self, i):
        while i // 2 > 0:
            # print(i // 2, i, len(self.heapList),i in self.handle)
            if not self.compare_function(self.heapList[i].item_,self.heapList[i // 2].item_):
                tmp = self.heapList[i // 2]
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
                self.handle[self.heapList[i // 2].handle_id_] = i//2
                self.handle[self.heapList[i].handle_id_] = i
            i = i // 2

    def __percDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.__minChild(i)
            if self.compare_function(self.heapList[i].item_, self.heapList[mc].item_):
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
                self.handle[self.heapList[i].handle_id_] = i
                self.handle[self.heapList[mc].handle_id_] = mc
            i = mc

    def __minChild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if not self.compare_function(self.heapList[i * 2].item_, self.heapList[i * 2 + 1].item_):
                return i * 2
            else:
                return i * 2 + 1

    def __len__(self):
        return self.size()

