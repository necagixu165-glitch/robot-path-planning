"""
机器人控制技术课程大作业
基于A*与RRT的路径规划对比实验
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random
import math
import time
import heapq
import platform

np.random.seed(42)
random.seed(42)

system = platform.system()
font_list = ['DejaVu Sans']
if system == 'Windows':
    for f in ['SimHei', 'Microsoft YaHei', 'SimSun']:
        if any(f.lower() in font.name.lower() for font in fm.fontManager.ttflist):
            font_list.insert(0, f)
            break
else:
    for f in ['Noto Sans CJK SC', 'WenQuanYi Zen Hei']:
        if any(f.lower() in font.name.lower() for font in fm.fontManager.ttflist):
            font_list.insert(0, f)
            break

plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False

try:
    fig_test, ax_test = plt.subplots(figsize=(1, 1))
    ax_test.set_title("测试")
    plt.close(fig_test)
    USE_CHINESE = True
except Exception:
    USE_CHINESE = False
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("[提示] 未找到中文字体，图表将使用英文标签")


def generate_map(size=50, obstacle_ratio=0.25, complex_mode=False):
    map_grid = np.zeros((size, size), dtype=int)
    map_grid[0, :] = 1
    map_grid[-1, :] = 1
    map_grid[:, 0] = 1
    map_grid[:, -1] = 1

    if complex_mode:
        map_grid[15, 5:25] = 1
        map_grid[15, 30:45] = 1
        map_grid[35, 10:30] = 1
        map_grid[35, 35:48] = 1
        map_grid[5:20, 25] = 1
        map_grid[22:35, 15] = 1
        map_grid[22:35, 40] = 1
        for _ in range(int(size * size * 0.05)):
            x, y = random.randint(1, size-2), random.randint(1, size-2)
            map_grid[x, y] = 1
    else:
        for _ in range(int(size * size * obstacle_ratio)):
            x, y = random.randint(1, size-2), random.randint(1, size-2)
            if not ((x < 5 and y < 5) or (x > size-5 and y > size-5)):
                map_grid[x, y] = 1

    return map_grid


class AStarPlanner:
    def __init__(self, map_grid):
        self.map = map_grid
        self.size = map_grid.shape[0]
        self.directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    def heuristic(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def plan(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        visited = set()
        nodes_expanded = 0

        while open_list:
            _, current = heapq.heappop(open_list)

            if current in visited:
                continue
            visited.add(current)
            nodes_expanded += 1

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path, nodes_expanded, len(visited)

            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)

                if 0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size:
                    if self.map[neighbor[0], neighbor[1]] == 1:
                        continue

                    move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                    tentative_g = g_score[current] + move_cost

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return None, nodes_expanded, len(visited)


class RRTPlanner:
    def __init__(self, map_grid, max_iter=5000, step_size=2.5):
        self.map = map_grid
        self.size = map_grid.shape[0]
        self.max_iter = max_iter
        self.step_size = step_size

    def check_collision(self, p1, p2):
        dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        steps = max(int(dist / 0.5), 1)

        for i in range(steps + 1):
            t = i / steps
            x = int(p1[0] + t * (p2[0]-p1[0]))
            y = int(p1[1] + t * (p2[1]-p1[1]))
            if 0 <= x < self.size and 0 <= y < self.size:
                if self.map[x, y] == 1:
                    return True
        return False

    def plan(self, start, goal):
        nodes = [start]
        parents = {start: None}

        for i in range(self.max_iter):
            if random.random() < 0.2:
                rand_point = goal
            else:
                rand_point = (random.randint(0, self.size-1), random.randint(0, self.size-1))

            nearest = min(nodes, key=lambda n: (n[0]-rand_point[0])**2 + (n[1]-rand_point[1])**2)

            dist = math.sqrt((nearest[0]-rand_point[0])**2 + (nearest[1]-rand_point[1])**2)
            if dist == 0:
                continue

            ratio = min(self.step_size / dist, 1.0)
            new_node = (
                int(nearest[0] + ratio * (rand_point[0]-nearest[0])),
                int(nearest[1] + ratio * (rand_point[1]-nearest[1]))
            )

            if new_node in parents:
                continue

            if not self.check_collision(nearest, new_node):
                nodes.append(new_node)
                parents[new_node] = nearest

                dist_to_goal = math.sqrt((new_node[0]-goal[0])**2 + (new_node[1]-goal[1])**2)
                if dist_to_goal < self.step_size:
                    if not self.check_collision(new_node, goal):
                        path = [goal]
                        node = new_node
                        while node is not None:
                            path.append(node)
                            node = parents[node]
                        path.reverse()
                        return path, i+1, len(nodes)

        return None, self.max_iter, len(nodes)


def calc_path_length(path):
    if path is None or len(path) < 2:
        return float('inf')
    length = 0.0
    for i in range(len(path)-1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        length += math.sqrt(dx*dx + dy*dy)
    return length


def calc_smoothness(path):
    if path is None or len(path) < 3:
        return 0.0
    angles = []
    for i in range(1, len(path)-1):
        v1 = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        v2 = (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        norm1 = math.sqrt(v1[0]**2 + v1[1]**2)
        norm2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if norm1 > 0 and norm2 > 0:
            cos_angle = max(min(dot/(norm1*norm2), 1.0), -1.0)
            angles.append(math.acos(cos_angle))
    return sum(angles) / len(angles) if angles else 0.0


def main():
    print("=" * 60)
    print("机器人控制技术课程大作业 —— 路径规划对比实验")
    print("=" * 60)

    results = []

    print("\n[实验1] 正在生成简单地图并运行算法...")
    map1 = generate_map(size=50, obstacle_ratio=0.25, complex_mode=False)
    start1 = (5, 5)
    goal1 = (44, 44)
    map1[start1[0], start1[1]] = 0
    map1[goal1[0], goal1[1]] = 0

    astar = AStarPlanner(map1)
    t0 = time.time()
    path_a1, expand_a1, visit_a1 = astar.plan(start1, goal1)
    t_a1 = (time.time() - t0) * 1000

    rrt = RRTPlanner(map1, max_iter=5000, step_size=3.0)
    t0 = time.time()
    path_r1, iter_r1, nodes_r1 = rrt.plan(start1, goal1)
    t_r1 = (time.time() - t0) * 1000

    results.append({
        'name': '简单地图' if USE_CHINESE else 'Simple Map',
        'map': map1, 'start': start1, 'goal': goal1,
        'astar': {
            'path': path_a1, 'time': t_a1, 'nodes': expand_a1,
            'length': calc_path_length(path_a1), 'smooth': calc_smoothness(path_a1)
        },
        'rrt': {
            'path': path_r1, 'time': t_r1, 'nodes': nodes_r1,
            'length': calc_path_length(path_r1), 'smooth': calc_smoothness(path_r1)
        }
    })
    print(f"  [完成] A*: 长度={results[-1]['astar']['length']:.2f}, 时间={t_a1:.2f}ms")
    print(f"  [完成] RRT: 长度={results[-1]['rrt']['length']:.2f}, 时间={t_r1:.2f}ms")

    print("\n[实验2] 正在生成复杂地图并运行算法...")
    map2 = generate_map(size=50, complex_mode=True)
    start2 = (3, 3)
    goal2 = (46, 46)
    map2[start2[0], start2[1]] = 0
    map2[goal2[0], goal2[1]] = 0

    astar2 = AStarPlanner(map2)
    t0 = time.time()
    path_a2, expand_a2, _ = astar2.plan(start2, goal2)
    t_a2 = (time.time() - t0) * 1000

    rrt2 = RRTPlanner(map2, max_iter=8000, step_size=2.5)
    t0 = time.time()
    path_r2, iter_r2, nodes_r2 = rrt2.plan(start2, goal2)
    t_r2 = (time.time() - t0) * 1000

    results.append({
        'name': '复杂地图' if USE_CHINESE else 'Complex Map',
        'map': map2, 'start': start2, 'goal': goal2,
        'astar': {
            'path': path_a2, 'time': t_a2, 'nodes': expand_a2,
            'length': calc_path_length(path_a2), 'smooth': calc_smoothness(path_a2)
        },
        'rrt': {
            'path': path_r2, 'time': t_r2, 'nodes': nodes_r2,
            'length': calc_path_length(path_r2), 'smooth': calc_smoothness(path_r2)
        }
    })
    print(f"  [完成] A*: 长度={results[-1]['astar']['length']:.2f}, 时间={t_a2:.2f}ms")
    print(f"  [完成] RRT: 长度={results[-1]['rrt']['length']:.2f}, 时间={t_r2:.2f}ms")

    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"{'场景':<10} {'算法':<6} {'路径长度':<10} {'时间(ms)':<10} {'节点数':<8} {'平滑度':<8}")
    print("-" * 60)
    for r in results:
        a = r['astar']
        rt = r['rrt']
        print(f"{r['name']:<10} {'A*':<6} {a['length']:<10.2f} {a['time']:<10.2f} {a['nodes']:<8} {a['smooth']:<8.3f}")
        print(f"{'':<10} {'RRT':<6} {rt['length']:<10.2f} {rt['time']:<10.2f} {rt['nodes']:<8} {rt['smooth']:<8.3f}")

    print("\n正在生成可视化图表...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, r in enumerate(results):
        ax = axes[0, idx]
        ax.imshow(r['map'], cmap='gray_r', origin='lower')

        if r['astar']['path']:
            px = [p[1] for p in r['astar']['path']]
            py = [p[0] for p in r['astar']['path']]
            ax.plot(px, py, 'b-', linewidth=2.5, label=f"A* ({r['astar']['length']:.1f})")

        if r['rrt']['path']:
            px = [p[1] for p in r['rrt']['path']]
            py = [p[0] for p in r['rrt']['path']]
            ax.plot(px, py, 'r-', linewidth=2, alpha=0.7, label=f"RRT ({r['rrt']['length']:.1f})")

        ax.plot(r['start'][1], r['start'][0], 'go', markersize=12, label='起点')
        ax.plot(r['goal'][1], r['goal'][0], 'y*', markersize=18, label='终点')
        title = f"{r['name']} - 路径对比" if USE_CHINESE else f"{r['name']} Path"
        ax.set_title(title, fontsize=13)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, [r['astar']['length'] for r in results], width, label='A*', color='steelblue')
    bars2 = ax.bar(x + width/2, [r['rrt']['length'] for r in results], width, label='RRT', color='coral')
    ylabel = '路径长度（栅格单位）' if USE_CHINESE else 'Path Length'
    title = '路径长度对比' if USE_CHINESE else 'Length Comparison'
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    ax = axes[1, 1]
    x = np.arange(len(names))
    width = 0.35
    ax2 = ax.twinx()
    ax.bar(x - width/2, [r['astar']['time'] for r in results], width, label='A* time', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, [r['rrt']['time'] for r in results], width, label='RRT time', color='coral', alpha=0.8)
    ax2.plot(x, [r['astar']['nodes'] for r in results], 'bo-', linewidth=2, markersize=8, label='A* nodes')
    ax2.plot(x, [r['rrt']['nodes'] for r in results], 'ro-', linewidth=2, markersize=8, label='RRT nodes')
    ax.set_ylabel('运行时间 (ms)' if USE_CHINESE else 'Time (ms)')
    ax2.set_ylabel('节点数量' if USE_CHINESE else 'Node Count')
    ax.set_title('运行时间与节点数对比' if USE_CHINESE else 'Time & Nodes')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('experiment_results（对照图片）.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n[完成] 结果图已保存为 experiment_results（对照图片）.png")
    print("=" * 60)


if __name__ == '__main__':
    main()
