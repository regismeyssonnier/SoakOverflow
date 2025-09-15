# -*- coding: latin-1 -*-
import pygame
import random
import time
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import time
from datetime import timedelta
from conv2d import *
from weights_unicodered import *
import matplotlib.pyplot as plt

# Constants couleurs
WHITE = (255, 255, 255)
GREY = (180, 180, 180)
DARK_GREY = (100, 100, 100)
BLACK = (0, 0, 0)

CELL_SIZE = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode_unicode_string_to_weights(unicode_str, offset=12.0, divider=2048.0, shape=None):
	# Étape 1 : reconstruire la chaîne binaire 'weights_bytes' comme en C++ wstring -> string
	weights_bytes = bytearray()
	for c in unicode_str:
		val = ord(c)
		weights_bytes.append((val >> 8) & 0xFF)  # octet haut
		weights_bytes.append(val & 0xFF)         # octet bas

	# Étape 2 : lire les poids 2 octets par 2 octets, big-endian
	size = len(weights_bytes) // 2
	output = []
	for i in range(size):
		s1 = weights_bytes[2*i]
		s2 = weights_bytes[2*i + 1]
		s = (s1 << 8) + s2
		val = (s / divider) - offset
		output.append(val)

	# Étape 3 : si shape précisé, reshape en numpy array
	if shape is not None:
		import numpy as np
		output = np.array(output, dtype=np.float32).reshape(shape)
	else:
		output = list(output)

	return output

import heapq

class Graph:
	def __init__(self, board):
		self.board = board
		self.height = self.board.height
		self.width = self.board.width
		self.n = self.height * self.width
		self.adj = [[] for _ in range(self.n)]
		self.distance = []

		self.create_graph()

	def create_graph(self):
		dx = [0, 1, 0, -1]
		dy = [-1, 0, 1, 0]

		for i in range(self.height):
			for j in range(self.width):
				cell = self.board.get(j, i)
				t = cell.get_type()
				if t != 0:
					continue

				from_node = i * self.width + j

				for k in range(4):
					x = j + dx[k]
					y = i + dy[k]

					if 0 <= x < self.width and 0 <= y < self.height:
						cell = self.board.get(x, y)
						t = cell.get_type()
						if t == 0:
							to_node = y * self.width + x
							self.adj[from_node].append((1, to_node))  # cost = 1

		for i in range(self.n):
			dist = [0] * self.n
			self.pathfinding(i, dist)
			self.distance.append(dist)

	def pathfinding(self, start, distance):
		n = self.n
		dist = [float('inf')] * n
		visited = [False] * n
		par = [-1] * n

		pq = []
		heapq.heappush(pq, (0, start))
		dist[start] = 0

		while pq:
			d, a = heapq.heappop(pq)

			if visited[a]:
				continue
			visited[a] = True

			for cost, b in self.adj[a]:
				if dist[a] + cost < dist[b]:
					dist[b] = dist[a] + cost
					par[b] = a
					heapq.heappush(pq, (dist[b], b))

		for i in range(n):
			distance[i] = dist[i]


#-------------------------------GRID-----------------------
import random
from collections import deque

import random
from collections import deque

class GridMaker:
	GRID_W_RATIO = 2
	MIN_SPAWN_COUNT = 3
	MAX_SPAWN_COUNT = 5
	MIN_HEIGHT = 6
	MAX_HEIGHT = 10

	@staticmethod
	def init_empty(w, h):
		grid = Grid(w, h, False)
		for y in range(h):
			for x in range(w):
				grid.get(x, y).set_type(Tile.TYPE_FLOOR)
		return grid

	@staticmethod
	def init_grid(rng):
		h = rng.randint(GridMaker.MIN_HEIGHT, GridMaker.MAX_HEIGHT)
		w = h * GridMaker.GRID_W_RATIO
		y_sym = rng.choice([True, False]) or rng.choice([True, False])
		grid = Grid(w, h, y_sym)
		all_coords = list(grid.get_coords())

		# Walls
		for y in range(1, h-1):
			for x in range(1, w//2 - 1):
				coord = Coord(x, y)
				opp = grid.opposite(coord)
				n = rng.randint(0, 9)
				type_ = Tile.TYPE_FLOOR
				if n == 0:
					type_ = Tile.TYPE_HIGH_COVER
				elif n == 1:
					type_ = Tile.TYPE_LOW_COVER
				grid.get(coord).set_type(type_)
				grid.get(opp).set_type(type_)

		# Spawns
		all_left_coords = [coord for coord in grid.get_coords() if coord.get_x() == 0]
		spawn_count = rng.randint(GridMaker.MIN_SPAWN_COUNT, GridMaker.MAX_SPAWN_COUNT)
		rng.shuffle(all_left_coords)
		if spawn_count == 5 and rng.choice([True, False]):
			spawn_count -= 1
		if spawn_count == 4 and rng.choice([True, False]):
			spawn_count -= 1
		for i in range(spawn_count):
			c = all_left_coords[i]
			grid.spawns.append(c)
			grid.get(c).clear()
			grid.get(grid.opposite(c)).clear()

		wall_coords = [c for c in grid.get_coords() if grid.get(c).is_cover()]
		GridMaker.fix_islands(grid, list(wall_coords), rng)
		return grid

	@staticmethod
	def get_island_from(islands, coord):
		for s in islands:
			if coord in s:
				return s
		return None

	@staticmethod
	def close_island_gapO(grid, wall_coords, islands):
		connecting_islands = None
		bridge = None
		for coord in wall_coords:
			neighs = grid.get_neighbours(coord)
			#connecting_islands = list({GridMaker.get_island_from(islands, n) for n in neighs if GridMaker.get_island_from(islands, n)})
			connecting_islands = set()

			for n in neighs:
				for idx, island in enumerate(islands):
					if n in island:
						connecting_islands.add(idx)
						break  # une coord n'appartient qu'� une ile

			if len(connecting_islands) > 1:
				bridge = coord
				break
		if bridge is not None:
			bridging = connecting_islands
			coord = bridge
			opposite = grid.opposite(coord)
			grid.get(coord).clear()
			grid.get(opposite).clear()
			wall_coords.remove(coord)
			wall_coords.remove(opposite)
			new_islands = [s for s in islands if s not in bridging]
			new_island = set()
			for s in bridging:
				new_island |= s
			islands.clear()
			islands.extend(new_islands)
			islands.append(new_island)
			return True
		return False

	@staticmethod
	def detect_islands(grid):
		islands = []
		computed = set()
		current = set()
		for p in grid.get_coords():
			if grid.get(p).is_cover():
				continue
			if p not in computed:
				fifo = deque()
				fifo.append(p)
				computed.add(p)
				while fifo:
					e = fifo.popleft()
					for delta in Grid.ADJACENCY:
						n = e.add(delta)
						cell = grid.get(n)
						if cell.is_valid() and not cell.is_cover() and n not in computed:
							fifo.append(n)
							computed.add(n)
					current.add(e)
				islands.append(set(current))
				current.clear()
		return islands

	@staticmethod
	def close_island_gap(grid, wall_coords, islands):
		bridge = None
		connecting_island_indices = None

		for coord in wall_coords:
			neighs = grid.get_neighbours(coord)
			current_island_indices = set()

			for n in neighs:
				for idx, island in enumerate(islands):
					if n in island:
						current_island_indices.add(idx)
						break  # une coord n'appartient qu'� une seule �le

			if len(current_island_indices) > 1:
				bridge = coord
				connecting_island_indices = current_island_indices
				break  # on a trouv� un pont possible

		if bridge is not None:
			# Nettoyer les cases du pont et de son oppos�
			coord = bridge
			opposite = grid.opposite(coord)
			grid.get(coord).clear()
			grid.get(opposite).clear()

			# Met � jour la liste des murs
			if coord in wall_coords:
				wall_coords.remove(coord)
			if opposite in wall_coords:
				wall_coords.remove(opposite)

			# Fusionner les �les concern�es
			new_island = set()
			new_islands = []

			for idx, island in enumerate(islands):
				if idx in connecting_island_indices:
					new_island |= island
				else:
					new_islands.append(island)

			# Mettre � jour la liste originale
			new_islands.append(new_island)
			islands.clear()
			islands.extend(new_islands)

			return True

		return False


	@staticmethod
	def fix_islands(grid, wall_coords, rng):
		rng.shuffle(wall_coords)
		islands = GridMaker.detect_islands(grid)
		while len(islands) > 1:
			closed = GridMaker.close_island_gap(grid, wall_coords, islands)
			if not closed:
				wall_adj = GridMaker.find_wall_adjacent_to_free_space(wall_coords, grid)
				if wall_adj is not None:
					coord = wall_adj
					opposite = grid.opposite(coord)
					grid.get(coord).clear()
					grid.get(opposite).clear()
					wall_coords.remove(coord)
					wall_coords.remove(opposite)
				islands = GridMaker.detect_islands(grid)

	@staticmethod
	def find_wall_adjacent_to_free_space(wall_coords, grid):
		for c in wall_coords:
			neighs = grid.get_neighbours(c)
			for n in neighs:
				if not grid.get(n).is_cover():
					return c
		return None


class Coord:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def euclidean_to(self, x, y):
		return ((x - self.x) ** 2 + (y - self.y) ** 2) ** 0.5

	def sqr_euclidean_to(self, x, y):
		return (x - self.x) ** 2 + (y - self.y) ** 2

	def add(self, x, y=None):
		if y is None:
			x, y = x.x, x.y
		return Coord(self.x + x, self.y + y)

	def __hash__(self):
		return hash((self.x, self.y))

	def __eq__(self, other):
		return isinstance(other, Coord) and self.x == other.x and self.y == other.y

	def __repr__(self):
		return f"({self.x}, {self.y})"

	def to_int_string(self):
		return f"{self.x} {self.y}"

	def get_x(self):
		return self.x

	def get_y(self):
		return self.y

	def manhattan_to(self, other):
		if isinstance(other, Coord):
			return abs(self.x - other.x) + abs(self.y - other.y)
		x, y = other
		return abs(self.x - x) + abs(self.y - y)

	def chebyshev_to(self, other):
		if isinstance(other, Coord):
			return max(abs(self.x - other.x), abs(self.y - other.y))
		x, y = other
		return max(abs(self.x - x), abs(self.y - y))


class Tile:
	TYPE_FLOOR = 0
	TYPE_LOW_COVER = 1
	TYPE_HIGH_COVER = 2

	def __init__(self, coord, type_=TYPE_FLOOR):
		self.coord = coord
		self.type = type_

	def set_type(self, type_):
		self.type = type_

	def get_type(self):
		return self.type

	def is_cover(self):
		return self.type != Tile.TYPE_FLOOR

	def get_cover_modifier(self):
		if self.type == Tile.TYPE_LOW_COVER:
			return 0.5
		elif self.type == Tile.TYPE_HIGH_COVER:
			return 0.25
		return 1

	def clear(self):
		self.type = Tile.TYPE_FLOOR

	def is_valid(self):
		# Should compare with a NO_TILE instance
		return True

Tile.NO_TILE = Tile(Coord(-1, -1), -1)


from collections import OrderedDict

class Grid:
	ADJACENCY = [Coord(0, -1), Coord(1, 0), Coord(0, 1), Coord(-1, 0)]

	def __init__(self, width, height, y_symetry=False):
		self.width = width
		self.height = height
		self.y_symetry = y_symetry
		self.cells = OrderedDict()
		self.spawns = []
		for y in range(height):
			for x in range(width):
				coord = Coord(x, y)
				self.cells[coord] = Tile(coord)

	def get(self, x, y=None):
		if y is None:
			coord = x
			return self.cells.get(coord, Tile.NO_TILE)
		return self.cells.get(Coord(x, y), Tile.NO_TILE)

	def get_neighbours(self, pos):
		neighs = []
		for delta in self.ADJACENCY:
			n = Coord(pos.x + delta.x, pos.y + delta.y)
			if self.get(n) != Tile.NO_TILE:
				neighs.append(n)
		return neighs

	def get_coords(self):
		return list(self.cells.keys())

	def opposite(self, c):
		return Coord(self.width - c.x - 1, self.height - c.y - 1 if self.y_symetry else c.y)

	def is_y_symetric(self):
		return self.y_symetry

	def wall_up(self):
		for t in self.cells.values():
			t.set_type(Tile.TYPE_HIGH_COVER)

#-------------------------END GRID -------------------------

class State:
	def __init__(self, parent=None, hit=None):
		self.score = 0.0
		self.ucb = 0.0
		self.n = 0.0
		self.child = []
		self.parent = parent
		self.hit = hit
		self.hit2 = None
		self.id = -1
		self.shx = -1
		self.shy = -1
		self.thx = -1
		self.thy = -1


class Hit:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def clone(self):
		return Hit(self.x, self.y)


class MCTS:
	def __init__(self):
		self.C = math.sqrt(0.5)
		self.time_limit = 50  # ms (� g�rer selon ton environnement)
		self.ROLLOUT_SZ = 5
		self.NODE = 0
		self.rand = random.Random()

	def selection(self, state):
		if not state.child:
			return state

		maxi = float('-inf')
		best = None

		for child in state.child:
			if child.n == 0:
				child.ucb = float('inf')
			else:
				exploitation = child.score / child.n
				logN = math.log(max(1, state.n))
				exploration = self.C * math.sqrt(logN / child.n)
				child.ucb = exploitation + exploration

			if child.ucb > maxi:
				maxi = child.ucb
				best = child

		return best

	def expand(self, state, hits):
		if state.child:
			return

		for hit in hits:
			new_state = State(state, hit)
			state.child.append(new_state)
			self.NODE += 1

	def backpropagation(self, state, score):
		node = state
		while node is not None:
			node.n += 1
			node.score += score
			node = node.parent

	
	def get_hits(self, game, x, y, indb):
		valid_neighbors = []
		H = game.grid.height
		W = game.grid.width

		dx = [1, -1, 0, 0, 0]
		dy = [0, 0, 1, -1, 0]


		my_agent = []
		opp_agent = []
		if indb == 'red':
			my_agent = game.red
			opp_agent = game.blue
		else:
			my_agent = game.blue
			opp_agent = game.red

		for d in range(4):
			nx = x + dx[d]
			ny = y + dy[d]

			if 0 <= nx < W and 0 <= ny < H:
				cell = game.grid.get(nx, ny)
				t = cell.get_type()
				if t != 0:
					continue

				occupied = False

				for ag in my_agent:
					if ag.coord.x == x and ag.coord.y == y:
						continue
					if ag.coord.x == nx and ag.coord.y == ny:
						occupied = True
						break

				if not occupied:
					for ag in opp_agent:
						if ag.coord.x == nx and ag.coord.y == ny:
							occupied = True
							break

				if not occupied:
					valid_neighbors.append(Hit(nx, ny))

		return valid_neighbors

	def get_FloorScore(self, game):

		my_count = 0
		opp_count = 0

		for y in range(game.grid.height):
			for x in range(game.grid.width):
				best_dist1 = float('inf')
				best_dist2 = float('inf')

				for a in game.red:
					da = abs(y - a.coord.y) + abs(x - a.coord.x)
					if a.wetness >= 50:
						da *= 2
					best_dist1 = min(best_dist1, da)

				for a in game.blue:
					da = abs(y - a.coord.y) + abs(x - a.coord.x)
					if a.wetness >= 50:
						da *= 2
					best_dist2 = min(best_dist2, da)

				if best_dist1 < best_dist2:
					my_count += 1
				elif best_dist2 < best_dist1:
					opp_count += 1

		r = my_count - opp_count
		if r > 0:
			game.rscore += r
		else:
			game.bscore += -r

		return r


	

	def PlayS(self, indb, game):
		count = 0
		start_time = time.time()

		dx = [1, -1, 0, 0, 0]
		dy = [0, 0, 1, -1, 0]

		my_agent = []
		opp_agent = []
		if indb == 'red':
			my_agent = game.red
			opp_agent = game.blue
		else:
			my_agent = game.blue
			opp_agent = game.red

		#print(f"indb: {indb}, len(my_agent): {len(my_agent)}, len(opp_agent): {len(opp_agent)}")


		root = []
		for a in my_agent:
			hp = Hit(a.coord.x, a.coord.y)
			state = State(None, hp)
			root.append(state)

			h = Hit(-1, -1)
			mind = 2_000_0000
			maxs = -2_000_000

			for o in opp_agent:
				d = game.graph.distance[hp.y * game.grid.width + hp.x][o.coord.y * game.grid.width + o.coord.x]
				sc = (6 - o.splash_bombs) * 100 - d
				if sc > maxs:
					maxs = sc
					mind = d
					h.x = o.coord.x
					h.y = o.coord.y

			root[-1].hit2 = h


		while True:
			elapsed = (time.time() - start_time) * 1000.0
			if elapsed > self.time_limit:
				break

			sim_game = game.Clone()  #  doit cloner le jeu
			node = []
			for r in root:
				node.append(r)

			my_agent = []
			opp_agent = []
			if indb == 'red':
				my_agent = sim_game.red
				opp_agent = sim_game.blue
			else:
				my_agent = sim_game.blue
				opp_agent = sim_game.red


			for depth in range(0, 2):
				
				for idx, a in enumerate(root):
					if my_agent[idx].wetness >= 100:continue

					if len(node[idx].child) == 0:
						hits = self.get_hits(sim_game, my_agent[idx].coord.x, my_agent[idx].coord.y, indb)
						#print(f"idx: {idx}, hits: {hits} {indb}")

						self.expand(node[idx], hits)

					if len(node[idx].child) == 0:continue
					node[idx] = self.selection(node[idx])
					node[idx].hit2 = root[idx].hit2

					my_agent[idx].coord = Coord(node[idx].hit.x, node[idx].hit.y)


			r = self.get_FloorScore(sim_game)
						
			score2 = 0
			if indb == 'red' and r > 0:score2 = r / 100
			if indb == 'blue' and r < 0:score2 = -r / 100

			for idx, ag in enumerate(root):
				
				agent = my_agent[idx]

				if agent.wetness >= 100:
					self.backpropagation(node[idx], -1.0)
					continue

				
				score, score3, score4 = 0.0, 0.0, 0.0

				sc = sim_game.rscore - sim_game.bscore
				if indb == 'blue':
					sc = -sc

				# Ajustement des poids selon la situation
				if sc > 100:
					alpha, beta, omega, theta, phi = 0.1, 0.0, 0.9, 0.0, 0.55
				elif agent.splash_bombs > 0 or sc < 100:
					alpha, beta, omega, theta, phi = 0.6, 0.25, 0.0, 0.15, 0.55
				elif agent.wetness < 40:
					alpha, beta, omega, theta, phi = 0.5, 0.2, 0.0, 0.3, 0.55
				else:
					alpha, beta, omega, theta, phi = 0.2, 0.25, 0.4, 0.15, 0.55

				
				# Calcul de la distance entre les positions
				d = game.graph.distance[
					node[idx].hit.y * sim_game.grid.width + node[idx].hit.x
				][
					node[idx].hit2.y * sim_game.grid.width + node[idx].hit2.x
				]

				score = 0.0 if d == float('inf') else max(0.0, min(1.0, (100 - d) / 100.0))

				# Calcul du "cover" : estimation de s�curit�
				cover = 0
				'''
					for j in range(4):
						edx = agent.coord.x + dx[j]
						edy = agent.coord.y + dy[j]

						if not (0 <= edx < sim_game.grid.width and 0 <= edy < sim_game.grid.height):
							continue

						cell = sim_game.grid.get(edx, edy)
						t = cell.get_type()

						if t > 0:
							counta = 0
							for a in opp_agent:
								if ((agent.coord.x < edx and a.coord.x > edx) or
									(agent.coord.x > edx and a.coord.x < edx) or
									(agent.coord.y < edy and a.coord.y > edy) or
									(agent.coord.y > edy and a.coord.y < edy)):
									counta += 1
							cover += t * counta
				'''
				score3 = 0# max(0.0, min(1.0, cover / 20.0))

				# P�nalit� d'espacement : �vite que les agents soient trop proches
				'''
					spacing_penalty = 0.0
					my_agents_list = my_agent
					for ii in range(len(my_agents_list)):
						for jj in range(ii + 1, len(my_agents_list)):
							a1, a2 = my_agents_list[ii], my_agents_list[jj]
							dist = abs(a1.coord.x - a2.coord.x) + abs(a1.coord.y - a2.coord.y)
							if dist < 2:
								spacing_penalty += 0.1 * (2 - dist)

				'''

				score4 = 0; #max(0.0, min(1.0, 1.0 - spacing_penalty))

				# Estimation des d�g�ts potentiels
				'''
				damage = 0.0
					for oa in opp_agent:
						dist_to_agent = sim_game.graph.distance[oa.coord.y * sim_game.grid.width + oa.coord.x][agent.coord.y * sim_game.grid.width + agent.coord.x]
						if dist_to_agent <= oa.optimalRange and oa.cooldown == 0:
							damage += oa.soakingPower

					if agent.wetness > 0:
						score5 = damage / (101.0 - float(agent.wetness))
					else:
						score5 = 1.0
				'''
				
				#score5 = max(0.0, min(1.0, score5))

				scoref = score
				#scoref = (score2 * alpha + score * beta + score3 * omega + score4 * theta) - score5 * phi
				
				self.backpropagation(node[idx], scoref)



			count += 1

		# 5. Choix du meilleur coup

		hits = []
		for idx, ag in enumerate(root):
			best = None
			best_score = -float('inf')
			id = -1
			for child in ag.child:
				if child.n > 0:
					avg = child.score / child.n
					if avg > best_score:
						best_score = avg
						best = child

			if best is not None:
				hits.append(best.hit)
			else:
				hits.append(None)

		return hits

#-----------------------GAME--------------------------------
class Player:
	def __init__(self, coord, team):
		self.coord = coord  # Un objet Coord
		self.team = team    # "red" ou "blue"
		self.last_coord = coord
		self.mx_cooldown = random.randint(1, 5)
		self.cooldown = 0
		self.splash_bombs = random.randint(0, 3)
		self.wetness = 0   
		self.optimalRange = random.randint(2, 8)
		self.soakingPower = random.randint(8, 32)
		self.score = 0
		self.dead = 0
		self.thx = -1
		self.thy = -1
		

	def move(self, c):
		self.last_coord = self.coord
		self.coord = c

	def back_move(self):
		self.coord = self.last_coord

	def __repr__(self):
		return f"Player({self.coord}, '{self.team}')"

	def clone(self):
		new_player = Player(Coord(self.coord.x, self.coord.y), self.team)
		new_player.last_coord = Coord(self.last_coord.x, self.last_coord.y)
		new_player.mx_cooldown = self.mx_cooldown
		new_player.cooldown = self.cooldown
		new_player.splash_bombs = self.splash_bombs
		new_player.wetness = self.wetness
		new_player.optimalRange = self.optimalRange
		new_player.soakingPower = self.soakingPower
		new_player.score = self.score
		new_player.dead = self.dead
		new_player.thx = self.thx
		new_player.thy = self.thy
		return new_player


class Game:
	def __init__(self, grid, red, blue):
		self.grid = grid
		self.red = red
		self.blue = blue
		self.rscore = 0
		self.bscore = 0
		self.reward = 0
		self.reward2 = 0
		self.action = []
		self.actionag = 0
		self.graph = Graph(self.grid)
		self.width = self.grid.width
		self.height = self.grid.height

	def Clone(self):

		rd = []
		for p in self.red:
			pp = p.clone()
			rd.append(pp)

		bl = []
		for p in self.blue:
			pp = p.clone()
			bl.append(pp)
		game = Game(self.grid, rd, bl)
		game.rscore = self.rscore
		game.bscore = self.bscore
		game.reward = self.reward
		game.reward2 = self.reward2
		game.graph = self.graph

		return game

	def get_Move(self, x, y):
		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1)]

		possible_moves = []
		occupied = set(p.coord for p in self.red + self.blue)
		origin = Coord(x, y)
		for d in directions:
			new_pos = origin.add(d)

			cell = self.grid.get(new_pos.get_x(), new_pos.get_y())

			t = cell.get_type()
			if t != Tile.TYPE_FLOOR: continue
			if new_pos in occupied: continue
			if 0 <= new_pos.x < self.grid.width and 0 <= new_pos.y < self.grid.height:
				possible_moves.append(new_pos)

		return possible_moves

	def get_MoveX(self, x, y):
		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1)]

		possible_moves = []
		occupied = set(p.coord for p in self.red + self.blue)
		origin = Coord(x, y)
		for idx, d in enumerate(directions):
			new_pos = origin.add(d)

			if not (0 <= new_pos.x < self.grid.width and 0 <= new_pos.y < self.grid.height):
				continue

			cell = self.grid.get(new_pos.get_x(), new_pos.get_y())
			t = cell.get_type()
			if t != Tile.TYPE_FLOOR:
				continue
			if new_pos in occupied:
				continue

			possible_moves.append((new_pos, idx))

		return possible_moves

	def get_MoveI(self, x, y):
		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1)]

		possible_moves = []
		occupied = set(p.coord for p in self.red + self.blue)
		origin = Coord(x, y)
		for idx, d in enumerate(directions):
			new_pos = origin.add(d)

			if not (0 <= new_pos.x < self.grid.width and 0 <= new_pos.y < self.grid.height):
				continue

			cell = self.grid.get(new_pos.get_x(), new_pos.get_y())
			t = cell.get_type()
			if t != Tile.TYPE_FLOOR:
				continue
			if new_pos in occupied:
				continue

			possible_moves.append(idx)

		return possible_moves

	def init_NN(self):

		self.nnz = PolicyNet()
		checkpoint = torch.load('checkpoint6.pth', weights_only=True)
		self.nnz.load_state_dict(checkpoint['model_state_dict'])

	def init_NNUS(self):

		self.nnz = PolicyNet()
		checkpoint = torch.load('checkpoint6uslim.pth', weights_only=True)
		self.nnz.load_state_dict(checkpoint['model_state_dict'])

	def init_NNUSN(self):
		self.nnz = PolicyNet_Numpy()
		load_pytorch_weights_into_numpy_model('checkpoint6uslim.pth', self.nnz)

	def set_bn_eval(bn_layer, weight, bias, running_mean, running_var, shape):
		bn_layer.gamma = decode_unicode_string_to_weights(weight, shape=shape).reshape(1, shape[0], 1, 1)
		bn_layer.beta = decode_unicode_string_to_weights(bias, shape=shape).reshape(1, shape[0], 1, 1)
		bn_layer.running_mean = decode_unicode_string_to_weights(running_mean, shape=shape).reshape(1, shape[0], 1, 1)
		bn_layer.running_var = decode_unicode_string_to_weights(running_var, shape=shape).reshape(1, shape[0], 1, 1)
		
	def load_batchnorm_eval_only(self, bn_layer, weight_str, bias_str, shape):
		C = shape[0]
		bn_layer.gamma = decode_unicode_string_to_weights(weight_str, shape=shape).reshape(1, C, 1, 1)
		bn_layer.beta = decode_unicode_string_to_weights(bias_str, shape=shape).reshape(1, C, 1, 1)
	
		# Fake running stats for eval mode
		bn_layer.running_mean = np.zeros((1, C, 1, 1))
		bn_layer.running_var = np.ones((1, C, 1, 1))
		


	def init_NNUSNW(self):
		self.nnz = PolicyNet_Numpy()

		# Conv1
		self.nnz.conv1.weight = decode_unicode_string_to_weights(conv1_weight, shape=conv1_weight_shape)
		self.nnz.conv1.bias = decode_unicode_string_to_weights(conv1_bias, shape=conv1_bias_shape)

		# Conv2
		self.nnz.conv2.weight = decode_unicode_string_to_weights(conv2_weight, shape=conv2_weight_shape)
		self.nnz.conv2.bias = decode_unicode_string_to_weights(conv2_bias, shape=conv2_bias_shape)
		

		# Conv3
		self.nnz.conv3.weight = decode_unicode_string_to_weights(conv3_weight, shape=conv3_weight_shape)
		self.nnz.conv3.bias = decode_unicode_string_to_weights(conv3_bias, shape=conv3_bias_shape)

		# Fully connected
		self.nnz.fc1.weight = decode_unicode_string_to_weights(fc1_weight, shape=fc1_weight_shape)
		self.nnz.fc1.bias = decode_unicode_string_to_weights(fc1_bias, shape=fc1_bias_shape)
	
		self.nnz.fc2.weight = decode_unicode_string_to_weights(fc2_weight, shape=fc2_weight_shape)
		self.nnz.fc2.bias = decode_unicode_string_to_weights(fc2_bias, shape=fc2_bias_shape)
		
		#self.nnz.fc3.weight = decode_unicode_string_to_weights(fc3_weight, shape=fc3_weight_shape)
		#self.nnz.fc3.bias = decode_unicode_string_to_weights(fc3_bias, shape=fc3_bias_shape)
		
		#self.load_batchnorm_eval_only(self.nnz.bn1, bn1_weight, bn1_bias, bn1_weight_shape)
		#self.load_batchnorm_eval_only(self.nnz.bn2, bn2_weight, bn2_bias, bn2_weight_shape)
		#self.load_batchnorm_eval_only(self.nnz.bn3, bn3_weight, bn3_bias, bn3_weight_shape)
		#self.load_batchnorm_eval_only(self.nnz.bn_fc1, bn_fc1_weight, bn_fc1_bias, bn_fc1_weight_shape)
		#self.load_batchnorm_eval_only(self.nnz.bn_fc2, bn_fc2_weight, bn_fc2_bias, bn_fc2_weight_shape)
				

	def get_best_zone_for_agent(self, agent: Player, my_agents: list[Player], opp_agents: list[Player], width: int, height: int):
		best_zones = []
		max_enemy_score = -1

		directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]

		for dy in range(-4, 5):
			for dx in range(-4, 5):
				cx = agent.coord.x + dx
				cy = agent.coord.y + dy

				if abs(dx) + abs(dy) > 4:
					continue

				if cx < 0 or cx >= width or cy < 0 or cy >= height:
					continue

				# V�rifie que cette case n�est pas trop proche d�un co�quipier (sauf soi-m�me)
				too_close_to_ally = False
				for ally in my_agents:
					if ally is agent:
						continue
					if abs(ally.coord.x - cx) <= 1 and abs(ally.coord.y - cy) <= 1:
						too_close_to_ally = True
						break

				if too_close_to_ally:
					continue

				adjacent_enemies = 0
				enemy_score = 0

				for dx_dir, dy_dir in directions:
					ex = cx + dx_dir
					ey = cy + dy_dir

					for opp in opp_agents:
						if opp.coord.x == ex and opp.coord.y == ey:
							adjacent_enemies += 1
							enemy_score += 10
							enemy_score += opp.splash_bombs * 10 + (opp.wetness + 30) * 1000
							break

				if adjacent_enemies > 0:
					if enemy_score > max_enemy_score:
						max_enemy_score = enemy_score
						best_zones = [(cx, cy)]
					elif enemy_score == max_enemy_score:
						best_zones.append((cx, cy))

		return best_zones


	def PlayX(self, ind=1):

		occupied = set(p.coord for p in self.red + self.blue)
		self.action = []

		if ind == 1:
			# Actions des rouges uniquement
			for p in self.red:
				#if p.wetness >= 100:
				#	self.action.append(4)  # action "immobile"
				#	continue
				poss_moves = self.get_MoveX(p.coord.get_x(), p.coord.get_y())
				if len(poss_moves) == 0:
					self.action.append(4)  # no move possible, action par d�faut
					continue
				mv, idx = random.choice(poss_moves)
				if mv not in occupied:
					p.move(mv)
					self.action.append(idx)
				else:
					self.action.append(4)

			for p in self.blue:
				poss_moves = self.get_Move(p.coord.get_x(), p.coord.get_y())
				if len(poss_moves) == 0:continue
				mv = random.choice(poss_moves)
				if mv not in occupied:
					p.move(mv)

			# Pour les bleus on n�ajoute rien dans self.action,
			# ou on ajoute une action neutre si tu veux toujours m�me taille

		elif ind == 2:
			for p in self.red:
				poss_moves = self.get_Move(p.coord.get_x(), p.coord.get_y())
				if len(poss_moves) == 0:continue
				mv = random.choice(poss_moves)
				if mv not in occupied:
					p.move(mv)


			# Actions des bleus uniquement
			for p in self.blue:
				#if p.wetness >= 100:
				#	self.action.append(4)
				#	continue
				poss_moves = self.get_MoveX(p.coord.get_x(), p.coord.get_y())
				if len(poss_moves) == 0:
					self.action.append(4)
					continue
				mv, idx = random.choice(poss_moves)
				if mv not in occupied:
					p.move(mv)
					self.action.append(idx)
				else:
					self.action.append(4)

		# Correction position conflictuelle / retour arri�re
		# Note: ici, self.action a autant d��l�ments que de joueurs concern�s (rouges ou bleus)
		players = self.red if ind == 1 else self.blue
		for idx, p in enumerate(players):
			if p.wetness >= 100:
				continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl != p)
			if p.coord in occupied:
				p.back_move()
				self.action[idx] = 4  # action "retour arri�re"

		players = self.red if ind == 2 else self.blue
		for p in players:
			occupied = set(pl.coord for pl in self.red + self.blue if p != pl)
			if p.coord in occupied:
				p.back_move()

		if(len(self.action) < 5):
			##print("ACTION=", len(self.action))
			while len(self.action) < 5:
				self.action.append(4)


		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		self.remove_wet_players()

		self.get_FloorScore()

		score = self.rscore - self.bscore

		self.reward = 0
		if ind == 1:
			if score > 0:
				self.reward = score

		if ind == 1:
			if score < 0:
				self.reward = -score
						
		##print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		#self.#print_wetness()

		if abs(score) >= 600:
			if score > 0:return 1
			elif score < 0:return -1
		
		if len(self.red) == 0:return -1
		if len(self.blue) == 0:return 1

		return -2

	def get_neighbors_around(self, cx, cy, players):
		neighbors = []
		for p in players:
			px, py = p.coord.x, p.coord.y
			if abs(px - cx) <= 1 and abs(py - cy) <= 1:
				if not (px == cx and py == cy):  # Exclure le centre
					neighbors.append(p)
		return neighbors


	def PlayX10(self, ind=1):

		occupied = set(p.coord for p in self.red + self.blue)
		self.action = []

		action2 = []

		
		# Actions des rouges uniquement
		for p in self.red:
			#if p.wetness >= 100:
			#	self.action.append(4)  # action "immobile"
			#	continue
			poss_moves = self.get_MoveX(p.coord.get_x(), p.coord.get_y())
			if len(poss_moves) == 0:
				self.action.append(4)  # no move possible, action par d�faut
				continue
			mv, idx = random.choice(poss_moves)
			if mv not in occupied:
				p.move(mv)
				self.action.append(idx)
			else:
				self.action.append(4)

		for idx, p in enumerate(self.red):
			#if p.wetness >= 100:
			#	continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl != p)
			if p.coord in occupied:
				p.back_move()
				self.action[idx] = 4  # action "retour arri�re"

		if(len(self.action) < 5):
			##print("ACTION=", len(self.action))
			while len(self.action) < 5:
				self.action.append(4)
	

		# Actions des bleus uniquement
		for p in self.blue:
			#if p.wetness >= 100:
			#	self.action.append(4)
			#	continue
			poss_moves = self.get_MoveX(p.coord.get_x(), p.coord.get_y())
			if len(poss_moves) == 0:
				action2.append(4)
				continue
			mv, idx = random.choice(poss_moves)
			if mv not in occupied:
				p.move(mv)
				action2.append(idx)
			else:
				action2.append(4)

	
		for idx, p in enumerate(self.blue):
			#if p.wetness >= 100:
			#	continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl != p)
			if p.coord in occupied:
				p.back_move()
				action2[idx] = 4  # action "retour arri�re"

		if(len(action2) < 5):
			##print("ACTION=", len(self.action))
			while len(action2) < 5:
				action2.append(4)

		
		self.action.extend(action2)
		
		#throw
		for p in self.red:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.red, self.blue, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30

				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		for p in self.blue:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.blue, self.red, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30
				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		#self.remove_wet_players()
		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		#self.remove_wet_players()

		self.get_FloorScore()

		score = self.rscore - self.bscore

		self.reward = 0
		self.reward2 = 0
		if score > 0:
			self.reward = score

		if score < 0:
			self.reward2 = -score
						
		##print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		#self.#print_wetness()

		if abs(score) >= 600:
			if score > 0:return 1
			elif score < 0:return -1
		
		if len(self.red) == 0:return -1
		if len(self.blue) == 0:return 1

		return -2

	def find_adjacent_free_spot(self, grid, x, y):
		for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
			nx, ny = x + dx, y + dy
			if 0 <= nx < grid.width and 0 <= ny < grid.height:
				if grid.get(nx, ny).get_type() == 0:
					return nx, ny
		return x, y  # si rien trouvé, on garde la case initiale


	def PlayX10Terr(self, ind=1):

		occupied = set(p.coord for p in self.red + self.blue)
		self.action = []

		action2 = []

		best_pos, _ = find_best_spot_numpy_general(self, "red")  # ou "blue" selon le camp
		best_x, best_y = best_pos
		best_x, best_y = self.find_adjacent_free_spot(self.grid, best_x, best_y)
		typec = self.grid.get(best_x, best_y).get_type()

		for p in self.red:
			# if p.wetness >= 100:
			#     self.action.append(4)  # immobile si trop mouillé
			#     continue

			poss_moves = self.get_MoveX(p.coord.get_x(), p.coord.get_y())
			if len(poss_moves) == 0:
				self.action.append(4)  # aucun mouvement possible
				continue

			# Choisir le mouvement qui minimise la distance à best_pos
			best_move = None
			best_idx = None
			min_dist = float('inf')

			for mv, idx in poss_moves:
				if mv in occupied:
					continue  # on évite les cases occupées

				if typec == 0:
					dist = self.graph.distance[mv.y * self.grid.width + mv.x][best_y * self.grid.width + best_x]
					
				else:
					dist = abs(mv.x - best_x) + abs(mv.y - best_y)  # distance de Manhattan

				if dist < min_dist:
					min_dist = dist
					best_move = mv
					best_idx = idx

			if best_move is not None:
				p.move(best_move)
				self.action.append(best_idx)
			else:
				self.action.append(4)  # aucune case libre vers best_pos



		

		if(len(self.action) < 5):
			##print("ACTION=", len(self.action))
			while len(self.action) < 5:
				self.action.append(4)
	

		occupied = set(p.coord for p in self.red + self.blue)

		# Actions des bleus uniquement
		best_pos, _ = find_best_spot_numpy_general(self, "blue")  # ou "blue" selon le camp
		best_x, best_y = best_pos
		best_x, best_y = self.find_adjacent_free_spot(self.grid, best_x, best_y)
		typec = self.grid.get(best_x, best_y).get_type()

		for p in self.blue:
			# if p.wetness >= 100:
			#     self.action.append(4)  # immobile si trop mouillé
			#     continue

			poss_moves = self.get_MoveX(p.coord.get_x(), p.coord.get_y())
			if len(poss_moves) == 0:
				action2.append(4)  # aucun mouvement possible
				continue

			# Choisir le mouvement qui minimise la distance à best_pos
			best_move = None
			best_idx = None
			min_dist = float('inf')

			for mv, idx in poss_moves:
				if mv in occupied:
					continue  # on évite les cases occupées

				if typec == 0:
					dist = self.graph.distance[mv.y * self.grid.width + mv.x][best_y * self.grid.width + best_x]
					
				else:
					dist = abs(mv.x - best_x) + abs(mv.y - best_y)  # distance de Manhattan

				if dist < min_dist:
					min_dist = dist
					best_move = mv
					best_idx = idx

			if best_move is not None:
				p.move(best_move)
				action2.append(best_idx)
			else:
				action2.append(4)  # aucune case libre vers best_pos


		for idx, p in enumerate(self.red):
			#if p.wetness >= 100:
			#	continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl.coord != p.coord)
			if p.coord in occupied:
				p.back_move()
				self.action[idx] = 4  # action "retour arri�re"
	
		for idx, p in enumerate(self.blue):
			#if p.wetness >= 100:
			#	continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl.coord != p.coord)
			if p.coord in occupied:
				p.back_move()
				action2[idx] = 4  # action "retour arri�re"

		if(len(action2) < 5):
			##print("ACTION=", len(self.action))
			while len(action2) < 5:
				action2.append(4)

		
		#self.action.extend(action2)
		if ind == 2:self.action = action2

		#throw
		for p in self.red:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.red, self.blue, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30

				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		for p in self.blue:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.blue, self.red, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30
				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		#self.remove_wet_players()
		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		#self.remove_wet_players()

		self.get_FloorScore()

		score = self.rscore - self.bscore

		self.reward = 0
		self.reward2 = 0
		if score > 0:
			self.reward = score

		if score < 0:
			self.reward2 = -score
						
		##print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		#self.#print_wetness()

		if abs(score) >= 600:
			if score > 0:return 1
			elif score < 0:return -1
		
		if len(self.red) == 0:return -1
		if len(self.blue) == 0:return 1

		return -2

	def PlayX10TerrMCTS(self, ind=1):

		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1), Coord(0,0)]

		occupied = set(p.coord for p in self.red + self.blue)
		self.action = []

		action2 = []
			

		mcts = MCTS()

		hits = mcts.PlayS('red', self.Clone())
		#print(hits)
		for idx, p in enumerate(self.red):
			if hits[idx] is None:
				self.action.append(4)
				continue
			mv = Coord(hits[idx].x, hits[idx].y)
			if mv not in occupied:
				delta = Coord(mv.x - p.coord.x, mv.y - p.coord.y)
								
				dir_index = directions.index(delta)
				self.action.append(dir_index)

				p.move(mv)
				
			else:
				self.action.append(4)

		mcts = MCTS()

		occupied = set(p.coord for p in self.red + self.blue)
		hits = mcts.PlayS('blue', self.Clone())
		#print(hits)
		for idx, p in enumerate(self.blue):
			if hits[idx] is None:
				action2.append(4)
				continue
			mv = Coord(hits[idx].x, hits[idx].y)
			if mv not in occupied:
				delta = Coord(mv.x - p.coord.x, mv.y - p.coord.y)
					
				dir_index = directions.index(delta)
				action2.append(dir_index)

				p.move(mv)
				
			else:
				action2.append(4)


		for idx, p in enumerate(self.red):
			#if p.wetness >= 100:
			#	continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl.coord != p.coord)
			if p.coord in occupied:
				p.back_move()
				self.action[idx] = 4  # action "retour arri�re"
	
		for idx, p in enumerate(self.blue):
			#if p.wetness >= 100:
			#	continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl.coord != p.coord)
			if p.coord in occupied:
				p.back_move()
				action2[idx] = 4  # action "retour arri�re"
									

		'''
			if(len(self.action) < 5):
				##print("ACTION=", len(self.action))
				while len(self.action) < 5:
					self.action.append(4)

			if(len(action2) < 5):
				##print("ACTION=", len(self.action))
				while len(action2) < 5:
					action2.append(4)
		'''

		if len(self.action) > 0:
			while len(self.action) < 5:
				for a in self.action[:]:
					if len(self.action) < 5:
						self.action.append(a)
					else:
						break
		else:
			self.action += [4] * 5

		if len(action2) > 0:
			while len(action2) < 5:
				for a in action2[:]:
					if len(action2) < 5:
						action2.append(a)
					else:
						break
		else:
			action2 += [4] * 5

		if ind == 2:self.action = action2
		
		#self.action.extend(action2)
		#print(self.action)
		
		#throw
		for p in self.red:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.red, self.blue, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30

				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		for p in self.blue:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.blue, self.red, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30
				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		#self.remove_wet_players()
		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		#self.remove_wet_players()

		self.get_FloorScore()

		score = self.rscore - self.bscore

		self.reward = 0
		self.reward2 = 0
		if score > 0:
			self.reward = score

		if score < 0:
			self.reward2 = -score
						
		##print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		#self.#print_wetness()

		if abs(score) >= 600:
			if score > 0:

				return 1
			elif score < 0:
			
				return -1
		
		if len(self.red) == 0:
			
			return -1
		if len(self.blue) == 0:
		
			return 1

		return -2

	
	def PlayX10TerrNN_vs_MCTS(self, ind, agent):

		ARG_MAX = False

		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1), Coord(0,0)]

		occupied = set(p.coord for p in self.red + self.blue)
		self.action = []

		action2 = []

		ACTION = []

		probs = 0
			
		player = self.red if ind == 1 else self.blue
		player2 = self.blue if ind == 1 else self.red
		# state_tensor_batch shape: (1, 83, 20, 10) par exemple
		
		actionag = 0

		##print("Actions pr�dites par joueur :", actions_list)
				

		if ind == 1:

			# Actions des rouges uniquement
			for i, p in enumerate(self.red):
				occupied = set(p.coord for p in self.red + self.blue)
				with torch.no_grad():
				
					poss_move = self.get_MoveI(p.coord.x, p.coord.y)
					mask = [False] * 5
					mask[4] = True
					for ip in poss_move:
						mask[ip] = True
					state_tensor = encode_ALL_RL(i, self.grid, self.red, self.blue, self)  # (93, 10, 20)
					input_tensor = state_tensor.to(dtype=torch.float32, device=device)  # (93, 10, 20)
					ACTION.append(agent.select_action(input_tensor, mask))  # select_action ajoute le batch
	
				if ACTION[i] == 4:
					continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[ACTION[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR:
					continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)
					
			mcts = MCTS()

			occupied = set(p.coord for p in self.red + self.blue)
			hits = mcts.PlayS('blue', self.Clone())
			#print(hits)
			for idx, p in enumerate(self.blue):
				if hits[idx] is None:
					action2.append(4)
					continue
				mv = Coord(hits[idx].x, hits[idx].y)
				if mv not in occupied:
					delta = Coord(mv.x - p.coord.x, mv.y - p.coord.y)
					
					dir_index = directions.index(delta)
					action2.append(dir_index)

					p.move(mv)
				
				else:
					action2.append(4)

			
		else:

			mcts = MCTS()

			hits = mcts.PlayS('red', self.Clone())
			#print(hits)
			for idx, p in enumerate(self.red):
				if hits[idx] is None:
					self.action.append(4)
					continue
				mv = Coord(hits[idx].x, hits[idx].y)
				if mv not in occupied:
					delta = Coord(mv.x - p.coord.x, mv.y - p.coord.y)
								
					dir_index = directions.index(delta)
					self.action.append(dir_index)

					p.move(mv)
				
				else:
					self.action.append(4)

			# Actions des rouges uniquement
			for i, p in enumerate(self.blue):
				occupied = set(p.coord for p in self.red + self.blue)
				with torch.no_grad():
				
					poss_move = self.get_MoveI(p.coord.x, p.coord.y)
					mask = [False] * 5
					mask[4] = True
					for ip in poss_move:
						mask[ip] = True
					state_tensor = encode_ALL_RL(i+len(self.red), self.grid, self.red, self.blue, self)  # (93, 10, 20)
					input_tensor = state_tensor.to(dtype=torch.float32, device=device)  # (93, 10, 20)
					ACTION.append(agent.select_action(input_tensor, mask))  # select_action ajoute le batch
	
				if ACTION[i] == 4:
					continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[ACTION[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR:
					continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)


		for idx, p in enumerate(self.red):
			#if p.wetness >= 100:
			#	continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl.coord != p.coord)
			if p.coord in occupied:
				p.back_move()
				self.action[idx] = 4  # action "retour arri�re"
	
		for idx, p in enumerate(self.blue):
			#if p.wetness >= 100:
			#	continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl.coord != p.coord)
			if p.coord in occupied:
				p.back_move()
				action2[idx] = 4  # action "retour arri�re"

		# stocker pour debug / replay
		self.actionag = ACTION

									
		'''
			if(len(self.action) < 5):
				##print("ACTION=", len(self.action))
				while len(self.action) < 5:
					self.action.append(4)

			if(len(action2) < 5):
				##print("ACTION=", len(self.action))
				while len(action2) < 5:
					action2.append(4)
		'''
		
		if ind == 2:self.action = action2
					
		#self.action.extend(action2)
		#print(self.action)
		
		#throw
		damage_r = 0
		for p in self.red:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.red, self.blue, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30

				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

			damage_r += p.wetness

		damage_b = 0
		for p in self.blue:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.blue, self.red, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30
				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

			damage_b += p.wetness

		#self.remove_wet_players()
		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		#self.remove_wet_players()

		self.get_FloorScore()

		score = self.rscore - self.bscore

		self.reward = 0
		self.reward2 = 0
		if score > 0:
			self.reward = self.rscore + damage_b * damage_b

		if score < 0:
			self.reward2 = self.bscore + damage_r * damage_r
						
		##print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		#self.#print_wetness()

		if abs(score) >= 600:
			if score > 0:
				
				return 1
			elif score < 0:
				
				return -1
		
		if len(self.red) == 0:
			
			return -1
		if len(self.blue) == 0:
			
			return 1

		return -2

	def PlayX10TerrNN_vs_Random(self, ind, agent):

		ARG_MAX = False

		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1), Coord(0,0)]

		occupied = set(p.coord for p in self.red + self.blue)
		self.action = []

		action2 = []

		ACTION = []

		probs = 0
			
		player = self.red if ind == 1 else self.blue
		player2 = self.blue if ind == 1 else self.red
		# state_tensor_batch shape: (1, 83, 20, 10) par exemple
		
		actionag = 0

		##print("Actions pr�dites par joueur :", actions_list)
				

		if ind == 1:

			# Actions des rouges uniquement
			for i, p in enumerate(self.red):
				occupied = set(p.coord for p in self.red + self.blue)
				with torch.no_grad():
				
					poss_move = self.get_MoveI(p.coord.x, p.coord.y)
					mask = [False] * 5
					mask[4] = True
					for ip in poss_move:
						mask[ip] = True
					state_tensor = encode_ALL_RL(i, self.grid, self.red, self.blue, self)  # (93, 10, 20)
					input_tensor = state_tensor.to(dtype=torch.float32, device=device)  # (93, 10, 20)
					ACTION.append(agent.select_action(input_tensor, mask))  # select_action ajoute le batch
	
				if ACTION[i] == 4:
					continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[ACTION[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR:
					continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)
					
			
			for p in self.blue:
				poss_moves = self.get_Move(p.coord.get_x(), p.coord.get_y())
				if len(poss_moves) == 0:continue
				mv = random.choice(poss_moves)
				if mv not in occupied:
					p.move(mv)

			
		else:

			for p in self.red:
				poss_moves = self.get_Move(p.coord.get_x(), p.coord.get_y())
				if len(poss_moves) == 0:continue
				mv = random.choice(poss_moves)
				if mv not in occupied:
					p.move(mv)


			# Actions des rouges uniquement
			for i, p in enumerate(self.blue):
				occupied = set(p.coord for p in self.red + self.blue)
				with torch.no_grad():
				
					poss_move = self.get_MoveI(p.coord.x, p.coord.y)
					mask = [False] * 5
					mask[4] = True
					for ip in poss_move:
						mask[ip] = True
					state_tensor = encode_ALL_RL(i+len(self.red), self.grid, self.red, self.blue, self)  # (93, 10, 20)
					input_tensor = state_tensor.to(dtype=torch.float32, device=device)  # (93, 10, 20)
					ACTION.append(agent.select_action(input_tensor, mask))  # select_action ajoute le batch
	
				if ACTION[i] == 4:
					continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[ACTION[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR:
					continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)


		for idx, p in enumerate(self.red):
			#if p.wetness >= 100:
			#	continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl.coord != p.coord)
			if p.coord in occupied:
				p.back_move()
				self.action[idx] = 4  # action "retour arri�re"
	
		for idx, p in enumerate(self.blue):
			#if p.wetness >= 100:
			#	continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl.coord != p.coord)
			if p.coord in occupied:
				p.back_move()
				action2[idx] = 4  # action "retour arri�re"

		# stocker pour debug / replay
		self.actionag = ACTION

									
		'''
			if(len(self.action) < 5):
				##print("ACTION=", len(self.action))
				while len(self.action) < 5:
					self.action.append(4)

			if(len(action2) < 5):
				##print("ACTION=", len(self.action))
				while len(action2) < 5:
					action2.append(4)
		'''
		
		if ind == 2:self.action = action2
					
		#self.action.extend(action2)
		#print(self.action)
		
		#throw
		damage_r = 0
		for p in self.red:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.red, self.blue, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30

				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

			damage_r += p.wetness

		damage_b = 0
		for p in self.blue:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.blue, self.red, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30
				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

			damage_b += p.wetness

		#self.remove_wet_players()
		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		#self.remove_wet_players()

		r = self.get_FloorScore()

		score = self.rscore - self.bscore

		self.reward = 0
		self.reward2 = 0
		#if score > 0:
		self.reward = self.rscore + r + damage_b * damage_b - (damage_r * damage_r) / 2

		#if score < 0:
		self.reward2 = self.bscore + r + damage_r * damage_r
						
		##print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		#self.#print_wetness()

		if abs(score) >= 600:
			if score > 0:
				
				return 1
			elif score < 0:
				
				return -1
		
		if len(self.red) == 0:
			
			return -1
		if len(self.blue) == 0:
			
			return 1

		return -2

	def PlayX10TerrNN_vs_NN(self, ind, agent):

		ARG_MAX = False

		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1), Coord(0,0)]

		occupied = set(p.coord for p in self.red + self.blue)
		self.action = []

		action2 = []

		ACTION = []
		ACTION2 = []

		probs = 0
			
		player = self.red if ind == 1 else self.blue
		player2 = self.blue if ind == 1 else self.red
		# state_tensor_batch shape: (1, 83, 20, 10) par exemple
		
		actionag = 0

		
				
				

				##print("Actions pr�dites par joueur :", actions_list)
				

		if ind == 1:

			# Actions des rouges uniquement
			for i, p in enumerate(self.red):
				occupied = set(p.coord for p in self.red + self.blue)
				with torch.no_grad():
				
					poss_move = self.get_MoveI(p.coord.x, p.coord.y)
					mask = [False] * 5
					mask[4] = True
					for ip in poss_move:
						mask[ip] = True
					state_tensor = encode_ALL_RL(i, self.grid, self.red, self.blue, self)  # (93, 10, 20)
					input_tensor = state_tensor.to(dtype=torch.float32, device=device)  # (93, 10, 20)
					ACTION.append(agent.select_action(input_tensor, mask))  # select_action ajoute le batch
	
				if ACTION[i] == 4:
					continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[ACTION[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR:
					continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)
					
			# Actions des rouges uniquement
			for i, p in enumerate(self.blue):
				occupied = set(p.coord for p in self.red + self.blue)
				with torch.no_grad():
				
					poss_move = self.get_MoveI(p.coord.x, p.coord.y)
					mask = [False] * 5
					mask[4] = True
					for ip in poss_move:
						mask[ip] = True
					state_tensor = encode_ALL_RL(i+len(self.red), self.grid, self.red, self.blue, self)  # (93, 10, 20)
					input_tensor = state_tensor.to(dtype=torch.float32, device=device)  # (93, 10, 20)
					ACTION2.append(agent.select_action(input_tensor, mask))  # select_action ajoute le batch
	
				if ACTION2[i] == 4:
					continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[ACTION2[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR:
					continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)

			
		else:

			for i, p in enumerate(self.red):
				occupied = set(p.coord for p in self.red + self.blue)
				with torch.no_grad():
				
					poss_move = self.get_MoveI(p.coord.x, p.coord.y)
					mask = [False] * 5
					mask[4] = True
					for ip in poss_move:
						mask[ip] = True
					state_tensor = encode_ALL_RL(i, self.grid, self.red, self.blue, self)  # (93, 10, 20)
					input_tensor = state_tensor.to(dtype=torch.float32, device=device)  # (93, 10, 20)
					ACTION2.append(agent.select_action(input_tensor, mask))  # select_action ajoute le batch
	
				if ACTION2[i] == 4:
					continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[ACTION2[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR:
					continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)

			# Actions des rouges uniquement
			for i, p in enumerate(self.blue):
				occupied = set(p.coord for p in self.red + self.blue)
				with torch.no_grad():
				
					poss_move = self.get_MoveI(p.coord.x, p.coord.y)
					mask = [False] * 5
					mask[4] = True
					for ip in poss_move:
						mask[ip] = True
					state_tensor = encode_ALL_RL(i+len(self.red), self.grid, self.red, self.blue, self)  # (93, 10, 20)
					input_tensor = state_tensor.to(dtype=torch.float32, device=device)  # (93, 10, 20)
					ACTION.append(agent.select_action(input_tensor, mask))  # select_action ajoute le batch
	
				if ACTION[i] == 4:
					continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[ACTION[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR:
					continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)


		for idx, p in enumerate(self.red):
			#if p.wetness >= 100:
			#	continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl.coord != p.coord)
			if p.coord in occupied:
				p.back_move()
				self.action[idx] = 4  # action "retour arri�re"
	
		for idx, p in enumerate(self.blue):
			#if p.wetness >= 100:
			#	continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl.coord != p.coord)
			if p.coord in occupied:
				p.back_move()
				action2[idx] = 4  # action "retour arri�re"

		# stocker pour debug / replay
		self.actionag = ACTION

									
		'''
			if(len(self.action) < 5):
				##print("ACTION=", len(self.action))
				while len(self.action) < 5:
					self.action.append(4)

			if(len(action2) < 5):
				##print("ACTION=", len(self.action))
				while len(action2) < 5:
					action2.append(4)
		'''
		
		if ind == 2:self.action = action2
					
		#self.action.extend(action2)
		#print(self.action)
		
		#throw
		damage_r = 0
		for p in self.red:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.red, self.blue, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30

				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

			damage_r += p.wetness

		damage_b = 0
		for p in self.blue:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.blue, self.red, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30
				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

			damage_b += p.wetness

		#self.remove_wet_players()
		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		#self.remove_wet_players()

		self.get_FloorScore()

		score = self.rscore - self.bscore

		self.reward = 0
		self.reward2 = 0
		if score > 0:
			self.reward = self.rscore + damage_b * damage_b

		if score < 0:
			self.reward2 = self.bscore + damage_r * damage_r
						
		##print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		#self.#print_wetness()

		if abs(score) >= 600:
			if score > 0:
				
				return 1
			elif score < 0:
				
				return -1
		
		if len(self.red) == 0:
			
			return -1
		if len(self.blue) == 0:
			
			return 1

		return -2

	def PlayX_NN(self, ind=1):

		ARG_MAX = True

		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1)]

		occupied = set(p.coord for p in self.red + self.blue)
		self.action = []

		if ind == 1:
			# state_tensor_batch shape: (1, 83, 20, 10) par exemple
			self.nnz.eval()
			with torch.no_grad():
				state_tensor = encode_ALL_RL(self.grid, self.red, self.blue)  # shape (canaux, H, W)
				# Pour le batch, on ajoute une dimension (batch=1)
				state_tensor_batch = state_tensor.unsqueeze(0)
				logits = self.nnz(state_tensor_batch)  # (1, num_players=5, num_actions=5)
				logits = logits.squeeze(0)  # (5, 5) pour chaque joueur

				if ARG_MAX:
					actions = torch.argmax(logits, dim=1)  # (5,)
					actions_list = actions.tolist()  # liste d'actions [a0, a1, ..., a4]
				else:
					probs = F.softmax(logits, dim=-1)  # (5, 5)
					actions = torch.multinomial(probs, num_samples=1).squeeze(1)
					actions_list = actions.tolist()

			#print("Actions pr�dites par joueur :", actions_list)

			# Actions des rouges uniquement
			for i, p in enumerate(self.red):
				if actions_list[i] == 4:continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[actions_list[i]])
				cell = grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR: continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)
				
			for p in self.blue:
				poss_moves = self.get_Move(p.coord.get_x(), p.coord.get_y())
				if len(poss_moves) == 0:continue
				mv = random.choice(poss_moves)
				if mv not in occupied:
					p.move(mv)


			# Pour les bleus on n�ajoute rien dans self.action,
			# ou on ajoute une action neutre si tu veux toujours m�me taille

		elif ind == 2:
			for p in self.red:
				poss_moves = self.get_Move(p.coord.get_x(), p.coord.get_y())
				if len(poss_moves) == 0:continue
				mv = random.choice(poss_moves)
				if mv not in occupied:
					p.move(mv)


			self.nnz.eval()
			with torch.no_grad():
				state_tensor = encode_ALL_RL(self.grid, self.blue, self.red)  # shape (canaux, H, W)
				# Pour le batch, on ajoute une dimension (batch=1)
				state_tensor_batch = state_tensor.unsqueeze(0)
				logits = self.nnz(state_tensor_batch)  # (1, num_players=5, num_actions=5)
				logits = logits.squeeze(0)  # (5, 5) pour chaque joueur

				if ARG_MAX:
					actions = torch.argmax(logits, dim=1)  # (5,)
					actions_list = actions.tolist()  # liste d'actions [a0, a1, ..., a4]
				else:
					probs = F.softmax(logits, dim=-1)  # (5, 5)
					actions = torch.multinomial(probs, num_samples=1).squeeze(1)
					actions_list = actions.tolist()

			#print("Actions pr�dites par joueur :", actions_list)

			# Actions des rouges uniquement
			for i, p in enumerate(self.blue):
				if actions_list[i] == 4:continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[actions_list[i]])
				cell = grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR: continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)


		# Correction position conflictuelle / retour arri�re
		# Note: ici, self.action a autant d��l�ments que de joueurs concern�s (rouges ou bleus)
		players = self.red if ind == 1 else self.blue
		for idx, p in enumerate(players):
			if p.wetness >= 100:
				continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl != p)
			if p.coord in occupied:
				p.back_move()
				
		players = self.red if ind == 2 else self.blue
		for p in players:
			occupied = set(pl.coord for pl in self.red + self.blue if p != pl)
			if p.coord in occupied:
				p.back_move()

	

		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		#self.remove_wet_players()

		self.get_FloorScore()

		score = self.rscore - self.bscore

		self.reward = 0
		if ind == 1:
			if score > 0:
				self.reward = score

		if ind == 2:
			if score < 0:
				self.reward = -score
						
		##print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		#self.#print_wetness()

		if abs(score) >= 600:
			if score > 0:return 1
			elif score < 0:return -1
		
		if len(self.red) == 0:return -1
		if len(self.blue) == 0:return 1

		return -2

	def PlayX_NN10(self, ind=1):

		ARG_MAX = False

		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1)]

		occupied = set(p.coord for p in self.red + self.blue)
		self.action = []

		
		# state_tensor_batch shape: (1, 83, 20, 10) par exemple
		self.nnz.eval()
		with torch.no_grad():
			state_tensor = encode_ALL_RL(self.grid, self.red, self.blue)  # shape (canaux, H, W)
			# Pour le batch, on ajoute une dimension (batch=1)
			state_tensor_batch = state_tensor.unsqueeze(0)
			logits = self.nnz(state_tensor_batch)  # (1, num_players=5, num_actions=5)
			logits = logits.squeeze(0)  # (5, 5) pour chaque joueur

			if ARG_MAX:
				actions = torch.argmax(logits, dim=1)  # (5,)
				actions_list = actions.tolist()  # liste d'actions [a0, a1, ..., a4]
			else:
				probs = F.softmax(logits, dim=-1)  # (5, 5)
				actions = torch.multinomial(probs, num_samples=1).squeeze(1)
				actions_list = actions.tolist()

			#print("Actions pr�dites par joueur :", actions_list)

		if ind == 1:
			
			# Actions des rouges uniquement
			for i, p in enumerate(self.red):
				if actions_list[i] == 4:continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[actions_list[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR: continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)
				
			for p in self.blue:
				poss_moves = self.get_Move(p.coord.get_x(), p.coord.get_y())
				if len(poss_moves) == 0:continue
				mv = random.choice(poss_moves)
				if mv not in occupied:
					p.move(mv)


			# Pour les bleus on n�ajoute rien dans self.action,
			# ou on ajoute une action neutre si tu veux toujours m�me taille

		elif ind == 2:
			for p in self.red:
				poss_moves = self.get_Move(p.coord.get_x(), p.coord.get_y())
				if len(poss_moves) == 0:continue
				mv = random.choice(poss_moves)
				if mv not in occupied:
					p.move(mv)


			# Actions des rouges uniquement
			for i, p in enumerate(self.blue):
				if actions_list[i+5] == 4:continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[actions_list[i+5]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR: continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)


		# Correction position conflictuelle / retour arri�re
		# Note: ici, self.action a autant d��l�ments que de joueurs concern�s (rouges ou bleus)
		players = self.red if ind == 1 else self.blue
		for idx, p in enumerate(players):
			if p.wetness >= 100:
				continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl != p)
			if p.coord in occupied:
				p.back_move()
				
		players = self.red if ind == 2 else self.blue
		for p in players:
			occupied = set(pl.coord for pl in self.red + self.blue if p != pl)
			if p.coord in occupied:
				p.back_move()

		#throw
		for p in self.red:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.red, self.blue, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30

				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		for p in self.blue:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.blue, self.red, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30
				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		self.remove_wet_players()

		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		self.remove_wet_players()

		self.get_FloorScore()

		score = self.rscore - self.bscore

		self.reward = 0
		if ind == 1:
			if score > 0:
				self.reward = score

		if ind == 2:
			if score < 0:
				self.reward = -score
						
		##print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		#self.#print_wetness()

		if abs(score) >= 600:
			if score > 0:return 1
			elif score < 0:return -1
		
		if len(self.red) == 0:return -1
		if len(self.blue) == 0:return 1

		return -2

	def PlayX_NN10N(self, ind=1):

		ARG_MAX = True

		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1)]

		occupied = set(p.coord for p in self.red + self.blue)
		self.action = []

		actions_list = []
		# state_tensor_batch shape: (1, 83, 20, 10) par exemple
		with torch.no_grad():
			# Choisir le joueur courant

			player = self.red if ind == 1 else self.blue
		
			actions_list = []

			for idx, pl in enumerate(player):
				# Récupérer les moves possibles depuis la grille
				poss_move = self.get_MoveI(pl.coord.x, pl.coord.y)  # ex: [0,2,4]
		
				# Construire un masque booléen
				mask = np.zeros(5, dtype=bool)
				mask[4] = True  # "ne rien faire" toujours valide
				for i in poss_move:
					mask[i] = True

				# Encoder l'état
				ip = 0
				state_tensor = encode_ALL_RL_numpy(idx, self.grid, self.red, self.blue, self)  # (C,H,W)
				state_tensor_batch = np.expand_dims(state_tensor, axis=0)  # (1,C,H,W)

				# Passage dans le réseau -> logits pour ce joueur
				logits = self.nnz.forward(state_tensor_batch)  # shape (1, num_actions)
				logits_np = np.squeeze(logits, axis=0)        # (num_actions,)

				# Appliquer le masque : on met -inf aux actions interdites
				logits_masked = np.where(mask, logits_np, -1e9)

				# Choisir la meilleure action valide
				action = int(np.argmax(logits_masked))
				actions_list.append(action)


			print("Actions predites par joueur :", actions_list)

		if ind == 1:
			
			# Actions des rouges uniquement
			for i, p in enumerate(self.red):
				if actions_list[i] == 4:continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[actions_list[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR: continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)
				
			for p in self.blue:
				poss_moves = self.get_Move(p.coord.get_x(), p.coord.get_y())
				if len(poss_moves) == 0:continue
				mv = random.choice(poss_moves)
				if mv not in occupied:
					p.move(mv)


			# Pour les bleus on n�ajoute rien dans self.action,
			# ou on ajoute une action neutre si tu veux toujours m�me taille

		elif ind == 2:
			for p in self.red:
				poss_moves = self.get_Move(p.coord.get_x(), p.coord.get_y())
				if len(poss_moves) == 0:continue
				mv = random.choice(poss_moves)
				if mv not in occupied:
					p.move(mv)


			# Actions des rouges uniquement
			for i, p in enumerate(self.blue):
				if actions_list[i] == 4:continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[actions_list[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR: continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)


		# Correction position conflictuelle / retour arri�re
		# Note: ici, self.action a autant d��l�ments que de joueurs concern�s (rouges ou bleus)
		players = self.red if ind == 1 else self.blue
		for idx, p in enumerate(players):
			if p.wetness >= 100:
				continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl != p)
			if p.coord in occupied:
				p.back_move()
				
		players = self.red if ind == 2 else self.blue
		for p in players:
			occupied = set(pl.coord for pl in self.red + self.blue if p != pl)
			if p.coord in occupied:
				p.back_move()

		#throw
		for p in self.red:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.red, self.blue, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30

				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		for p in self.blue:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.blue, self.red, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30
				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		self.remove_wet_players()

		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		self.remove_wet_players()

		self.get_FloorScore()

		score = self.rscore - self.bscore

		self.reward = 0
		if ind == 1:
			if score > 0:
				self.reward = score

		if ind == 2:
			if score < 0:
				self.reward = -score
						
		##print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		#self.#print_wetness()

		if abs(score) >= 600:
			if score > 0:return 1
			elif score < 0:return -1
		
		if len(self.red) == 0:return -1
		if len(self.blue) == 0:return 1

		return -2
	
	

	def PlayX_NN10NMCTS(self, ind=1):

		ARG_MAX = False

		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1)]

		occupied = set(p.coord for p in self.red + self.blue)
		self.action = []

		
		# state_tensor_batch shape: (1, 83, 20, 10) par exemple
		self.nnz.eval()
		with torch.no_grad():
			player = self.red if ind == 1 else self.blue
			player2 = self.blue if ind == 1 else self.red
			state_tensor = encode_ALL_RL_numpy(self.grid, player, player2, self)  # (canaux, H, W)

			# Ajouter une dimension batch au début : shape devient (1, canaux, H, W)
			state_tensor_batch = np.expand_dims(state_tensor, axis=0)

			# Passage dans le réseau numpy
			logits = self.nnz.forward(state_tensor_batch)  # shape (1, num_players, num_actions)

			# Supprimer la dimension batch pour avoir (num_players, num_actions)
			logits = np.squeeze(logits, axis=0)


			if ARG_MAX:
				actions = torch.argmax(logits, dim=1)  # (5,)
				actions_list = actions.tolist()  # liste d'actions [a0, a1, ..., a4]
			else:
				probs = softmax(logits, axis=-1)
				actions = multinomial_numpy(probs)
				actions_list = actions.tolist()

			##print("Actions pr�dites par joueur :", actions_list)

		if ind == 1:

			# Actions des rouges uniquement
			for i, p in enumerate(self.red):
				if actions_list[i] == 4:continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[actions_list[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR: continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)

			mcts = MCTS()

			hits = mcts.PlayS('blue', self.Clone())
			#print(hits)
			for idx, p in enumerate(self.blue):
				if hits[idx] is None:continue
				mv = Coord(hits[idx].x, hits[idx].y)
				if mv not in occupied:
					p.move(mv)

			
			


			# Pour les bleus on n�ajoute rien dans self.action,
			# ou on ajoute une action neutre si tu veux toujours m�me taille

		elif ind == 2:
			
			mcts = MCTS()

			hits = mcts.PlayS('red', self.Clone())
			for idx, p in enumerate(self.red):
				if hits[idx] is None:continue
				mv = Coord(hits[idx].x, hits[idx].y)
				if mv not in occupied:
					p.move(mv)
			
			# Actions des rouges uniquement
			for i, p in enumerate(self.blue):
				if actions_list[i] == 4:continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[actions_list[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR: continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)
				


		# Correction position conflictuelle / retour arri�re
		# Note: ici, self.action a autant d��l�ments que de joueurs concern�s (rouges ou bleus)
		players = self.red if ind == 1 else self.blue
		for idx, p in enumerate(players):
			if p.wetness >= 100:
				continue
			occupied = set(pl.coord for pl in self.red + self.blue if pl.coord != p.coord)
			if p.coord in occupied:
				p.back_move()
				
		players = self.red if ind == 2 else self.blue
		for p in players:
			occupied = set(pl.coord for pl in self.red + self.blue if p.coord != pl.coord)
			if p.coord in occupied:
				p.back_move()

		#throw
		for p in self.red:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.red, self.blue, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30

				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		for p in self.blue:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.blue, self.red, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30
				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		self.remove_wet_players()

		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		self.remove_wet_players()

		self.get_FloorScore()

		score = self.rscore - self.bscore

		self.reward = 0
		if ind == 1:
			if score > 0:
				self.reward = score

		if ind == 2:
			if score < 0:
				self.reward = -score
						
		##print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		#self.#print_wetness()

		if abs(score) >= 600:
			if score > 0:return 1
			elif score < 0:return -1
		
		if len(self.red) == 0:return -1
		if len(self.blue) == 0:return 1

		return -2


	def PlayX_NN10AH(self, policy):

		ARG_MAX = False

		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1)]

		occupied = set(p.coord for p in self.red + self.blue)

		self.action = []

		action2 = []

		
		# state_tensor_batch shape: (1, 83, 20, 10) par exemple
		policy.eval()
		with torch.no_grad():
			state_tensor = encode_ALL_RL(self.grid, self.red, self.blue)  # shape (canaux, H, W)
			# Pour le batch, on ajoute une dimension (batch=1)
			state_tensor_batch = state_tensor.unsqueeze(0)
			logits = policy(state_tensor_batch)  # (1, num_players=5, num_actions=5)
			logits = logits.squeeze(0)  # (5, 5) pour chaque joueur

			if ARG_MAX:
				actions = torch.argmax(logits, dim=1)  # (5,)
				actions_list = actions.tolist()  # liste d'actions [a0, a1, ..., a4]
			else:
				probs = F.softmax(logits, dim=-1)  # (5, 5)
				actions = torch.multinomial(probs, num_samples=1).squeeze(1)
				actions_list = actions.tolist()

			##print("Actions pr�dites par joueur :", actions_list)

		
			
		# Actions des rouges uniquement
		for i, p in enumerate(self.red):
			if actions_list[i] == 4:
				self.action.append(4)
				continue
			origin = Coord(p.coord.x, p.coord.y)
			mv = origin.add(directions[actions_list[i]])
			cell = self.grid.get(mv.x, mv.y)
			t = cell.get_type()
			if t != Tile.TYPE_FLOOR: continue
			if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
				p.move(mv)
				self.action.append(actions_list[i])
			else:
				self.action.append(4)

		if(len(self.action) < 5):
			##print("ACTION=", len(self.action))
			while len(self.action) < 5:
				self.action.append(4)

		# Actions des rouges uniquement
		for i, p in enumerate(self.blue):
			if actions_list[i+5] == 4:
				action2.append(4)
				continue
			origin = Coord(p.coord.x, p.coord.y)
			mv = origin.add(directions[actions_list[i+5]])
			cell = self.grid.get(mv.x, mv.y)
			t = cell.get_type()
			if t != Tile.TYPE_FLOOR: continue
			if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
				p.move(mv)
				action2.append(actions_list[i+5])
			else:
				action2.append(4)

		if(len(action2) < 5):
			##print("ACTION=", len(self.action))
			while len(action2) < 5:
				action2.append(4)

		

		# Correction position conflictuelle / retour arri�re
		# Note: ici, self.action a autant d��l�ments que de joueurs concern�s (rouges ou bleus)
		for idx, p in enumerate(self.red):
			occupied = set(pl.coord for pl in self.red + self.blue if p != pl)
			if p.coord in occupied:
				p.back_move()
				self.action[idx] = 4
					
		for idx, p in enumerate(self.blue):
			occupied = set(pl.coord for pl in self.red + self.blue if p != pl)
			if p.coord in occupied:
				p.back_move()
				action2[idx] = 4

		self.action.extend(action2)

		#throw
		for p in self.red:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.red, self.blue, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30

				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		for p in self.blue:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.blue, self.red, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30
				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		#self.remove_wet_players()

		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		#self.remove_wet_players()

		self.get_FloorScore()

		score = self.rscore - self.bscore

		self.reward = 0
		self.reward2 = 0
		if score > 0:
			self.reward = score
		
		if score < 0:
			self.reward2 = -score
						
		##print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		#self.#print_wetness()

		if abs(score) >= 600:
			if score > 0:
				self.reward = 10000
				self.reward2 = -10000
				return 1
			elif score < 0:
				self.reward = -10000
				self.reward2 = 10000
				return -1
		
		if len(self.red) == 0:
			self.reward = -10000
			self.reward2 = 10000
			return -1
		if len(self.blue) == 0:
			self.reward = 10000
			self.reward2 = -10000
			return 1

		return -2

	def PlayX_NN10AH5(self, policy, ind):

		ARG_MAX = False

		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1)]

		occupied = set(p.coord for p in self.red + self.blue)

		self.action = []

		action2 = []

		player = self.red if ind == 1 else self.blue
		player2 = self.blue if ind == 1 else self.red
		# state_tensor_batch shape: (1, 83, 20, 10) par exemple
		policy.eval()
		with torch.no_grad():
			state_tensor = encode_ALL_RL(self.grid, player, player2, self)  # shape (canaux, H, W)
			# Pour le batch, on ajoute une dimension (batch=1)
			state_tensor_batch = state_tensor.unsqueeze(0)
			logits = policy(state_tensor_batch)  # (1, num_players=5, num_actions=5)
			logits = logits.squeeze(0)  # (5, 5) pour chaque joueur

			if ARG_MAX:
				actions = torch.argmax(logits, dim=1)  # (5,)
				actions_list = actions.tolist()  # liste d'actions [a0, a1, ..., a4]
			else:
				probs = F.softmax(logits, dim=-1)  # (5, 5)
				actions = torch.multinomial(probs, num_samples=1).squeeze(1)
				actions_list = actions.tolist()

			##print("Actions pr�dites par joueur :", actions_list)


		if ind == 1:
		
			
			# Actions des rouges uniquement
			for i, p in enumerate(self.red):
				if actions_list[i] == 4:
					self.action.append(4)
					continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[actions_list[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR: continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)
					self.action.append(actions_list[i])
				else:
					self.action.append(4)

			if(len(self.action) < 5):
				##print("ACTION=", len(self.action))
				while len(self.action) < 5:
					self.action.append(4)

			
			mcts = MCTS()

			hits = mcts.PlayS('blue', self.Clone())
			#print(hits)
			for idx, p in enumerate(self.blue):
				if hits[idx] is None:
					continue
				mv = Coord(hits[idx].x, hits[idx].y)
				if mv not in occupied:
					p.move(mv)
				
			
		else:

			
			mcts = MCTS()

			hits = mcts.PlayS('red', self.Clone())
			#print(hits)
			for idx, p in enumerate(self.red):
				if hits[idx] is None:
				
					continue
				mv = Coord(hits[idx].x, hits[idx].y)
				if mv not in occupied:
					
					p.move(mv)
				
	

			# Actions des rouges uniquement
			for i, p in enumerate(self.blue):
				if actions_list[i] == 4:
					action2.append(4)
					continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[actions_list[i]])
				cell = self.grid.get(mv.x, mv.y)
				t = cell.get_type()
				if t != Tile.TYPE_FLOOR: continue
				if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
					p.move(mv)
					action2.append(actions_list[i])
				else:
					action2.append(4)

			if(len(action2) < 5):
				##print("ACTION=", len(self.action))
				while len(action2) < 5:
					action2.append(4)

		

		# Correction position conflictuelle / retour arri�re
		# Note: ici, self.action a autant d��l�ments que de joueurs concern�s (rouges ou bleus)
		for idx, p in enumerate(self.red):
			occupied = set(pl.coord for pl in self.red + self.blue if p != pl)
			if p.coord in occupied:
				p.back_move()
				if ind == 1:self.action[idx] = 4
					
		for idx, p in enumerate(self.blue):
			occupied = set(pl.coord for pl in self.red + self.blue if p != pl)
			if p.coord in occupied:
				p.back_move()
				if ind == 2:action2[idx] = 4

		#self.action.extend(action2)
		if ind == 2:self.action = action2

		#throw
		for p in self.red:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.red, self.blue, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30

				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		for p in self.blue:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.blue, self.red, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30
				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		#self.remove_wet_players()

		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		#self.remove_wet_players()

		self.get_FloorScore()

		score = self.rscore - self.bscore

		self.reward = 0
		self.reward2 = 0
		if score > 0:
			self.reward = score
		
		if score < 0:
			self.reward2 = -score
						
		##print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		#self.#print_wetness()

		if abs(score) >= 600:
			if score > 0:
				self.reward = 10000
				self.reward2 = -10000
				return 1
			elif score < 0:
				self.reward = -10000
				self.reward2 = 10000
				return -1
		
		if len(self.red) == 0:
			self.reward = -10000
			self.reward2 = 10000
			return -1
		if len(self.blue) == 0:
			self.reward = 10000
			self.reward2 = -10000
			return 1

		return -2

	def PlayX_NN10AH52(self, policy, ind):

		ARG_MAX = False

		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1)]

		occupied = set(p.coord for p in self.red + self.blue)

		self.action = []

		action2 = []

		player = self.red if ind == 1 else self.blue
		player2 = self.blue if ind == 1 else self.red
		# state_tensor_batch shape: (1, 83, 20, 10) par exemple
		policy.eval()
		with torch.no_grad():
			state_tensor = encode_ALL_RL(self.grid, player, player2, self)  # shape (canaux, H, W)
			# Pour le batch, on ajoute une dimension (batch=1)
			state_tensor_batch = state_tensor.unsqueeze(0)
			logits = policy(state_tensor_batch)  # (1, num_players=5, num_actions=5)
			logits = logits.squeeze(0)  # (5, 5) pour chaque joueur

			if ARG_MAX:
				actions = torch.argmax(logits, dim=1)  # (5,)
				actions_list = actions.tolist()  # liste d'actions [a0, a1, ..., a4]
			else:
				probs = F.softmax(logits, dim=-1)  # (5, 5)
				actions = torch.multinomial(probs, num_samples=1).squeeze(1)
				actions_list = actions.tolist()

			##print("Actions pr�dites par joueur :", actions_list)

		with torch.no_grad():
			state_tensor = encode_ALL_RL(self.grid, player2, player, self)  # shape (canaux, H, W)
			# Pour le batch, on ajoute une dimension (batch=1)
			state_tensor_batch = state_tensor.unsqueeze(0)
			logits = policy(state_tensor_batch)  # (1, num_players=5, num_actions=5)
			logits = logits.squeeze(0)  # (5, 5) pour chaque joueur

			if ARG_MAX:
				actions = torch.argmax(logits, dim=1)  # (5,)
				actions_list2 = actions.tolist()  # liste d'actions [a0, a1, ..., a4]
			else:
				probs = F.softmax(logits, dim=-1)  # (5, 5)
				actions = torch.multinomial(probs, num_samples=1).squeeze(1)
				actions_list2 = actions.tolist()

			
		# Actions des rouges uniquement
		for i, p in enumerate(self.red):
			if actions_list[i] == 4:
				self.action.append(4)
				continue
			origin = Coord(p.coord.x, p.coord.y)
			mv = origin.add(directions[actions_list[i]])
			cell = self.grid.get(mv.x, mv.y)
			t = cell.get_type()
			if t != Tile.TYPE_FLOOR: continue
			if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
				p.move(mv)
				self.action.append(actions_list[i])
			else:
				self.action.append(4)

		if(len(self.action) < 5):
			##print("ACTION=", len(self.action))
			while len(self.action) < 5:
				self.action.append(4)

			
		# Actions des rouges uniquement
		for i, p in enumerate(self.blue):
			if actions_list[i] == 4:
				action2.append(4)
				continue
			origin = Coord(p.coord.x, p.coord.y)
			mv = origin.add(directions[actions_list[i]])
			cell = self.grid.get(mv.x, mv.y)
			t = cell.get_type()
			if t != Tile.TYPE_FLOOR: continue
			if mv not in occupied and mv.x >= 0 and mv.x < self.grid.width and mv.y >= 0 and mv.y < self.grid.height:
				p.move(mv)
				action2.append(actions_list[i])
			else:
				action2.append(4)

		if(len(action2) < 5):
			##print("ACTION=", len(self.action))
			while len(action2) < 5:
				action2.append(4)
				

		

		# Correction position conflictuelle / retour arri�re
		# Note: ici, self.action a autant d��l�ments que de joueurs concern�s (rouges ou bleus)
		for idx, p in enumerate(self.red):
			occupied = set(pl.coord for pl in self.red + self.blue if p.coord != pl.coord)
			if p.coord in occupied:
				p.back_move()
				if ind == 1:self.action[idx] = 4
					
		for idx, p in enumerate(self.blue):
			occupied = set(pl.coord for pl in self.red + self.blue if p.coord != pl.coord)
			if p.coord in occupied:
				p.back_move()
				if ind == 2:action2[idx] = 4

		#self.action.extend(action2)
		if ind == 2:self.action = action2

		#throw
		for p in self.red:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.red, self.blue, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30

				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		for p in self.blue:
			if p.splash_bombs > 0:
				zones = self.get_best_zone_for_agent(p, self.blue, self.red, width=self.grid.width, height=self.grid.height)
				if len(zones) > 0:
					p.thx, p.thy = zones[0]
					p.splash_bombs-= 1

					players_in_zone = self.get_neighbors_around(p.thx, p.thy, self.red + self.blue)
					for p in players_in_zone:
						p.wetness += 30
				else:
					p.txh, p.thy = -1, -1
			else:
				p.txh, p.thy = -1, -1

		#self.remove_wet_players()

		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		#self.remove_wet_players()

		self.get_FloorScore()

		score = self.rscore - self.bscore

		self.reward = 0
		self.reward2 = 0
		if score > 0:
			self.reward = score
		
		if score < 0:
			self.reward2 = -score
						
		##print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		#self.#print_wetness()

		if abs(score) >= 600:
			if score > 0:
				self.reward = 10000
				self.reward2 = -10000
				return 1
			elif score < 0:
				self.reward = -10000
				self.reward2 = 10000
				return -1
		
		if len(self.red) == 0:
			self.reward = -10000
			self.reward2 = 10000
			return -1
		if len(self.blue) == 0:
			self.reward = 10000
			self.reward2 = -10000
			return 1

		return -2



	def Play(self):

		occupied = set(p.coord for p in self.red + self.blue)

		for p in self.red:
			poss_moves = self.get_Move(p.coord.get_x(), p.coord.get_y())
			if len(poss_moves) == 0:continue
			mv = random.choice(poss_moves)
			if mv not in occupied:
				p.move(mv)

		for p in self.blue:
			poss_moves = self.get_Move(p.coord.get_x(), p.coord.get_y())
			if len(poss_moves) == 0:continue
			mv = random.choice(poss_moves)
			if mv not in occupied:
				p.move(mv)

		
		for p in self.red:
			occupied = set(pl.coord for pl in self.red + self.blue if p.coord != pl.coord)
			if p.coord in occupied:
				p.back_move()

		for p in self.blue:
			occupied = set(pl.coord for pl in self.red + self.blue if p.coord != pl.coord)
			if p.coord in occupied:
				p.back_move()


		#shoot
		self.Shoot(True)
		self.Shoot(False)
		self.Cooldown()
		self.remove_wet_players()

		self.get_FloorScore()

		score = self.rscore - self.bscore

		
		
		#print(f"Red Score: {self.rscore} | Blue Score: {self.bscore} | Diff: {self.rscore - self.bscore}")
		self.print_wetness()

		if abs(score) >= 600:
			if score > 0:return 1
			elif score < 0:return -1
		
		if len(self.red) == 0:return -1
		if len(self.blue) == 0:return 1

		return -2

	def Cooldown(self):

		for p in self.red+self.blue:
			if p.cooldown > 0:p.cooldown -= 1
			
			   
	def print_wetness(self):
		print("=== Red Players Wetness ===")
		for i, p in enumerate(self.red):
			print(f"Red {i}: wetness = {p.wetness}")

		print("=== Blue Players Wetness ===")
		for i, p in enumerate(self.blue):
			print(f"Blue {i}: wetness = {p.wetness}")



	def remove_wet_players(self):
		self.red = [p for p in self.red if p.wetness < 100]
		self.blue = [p for p in self.blue if p.wetness < 100]

	def Shoot(self, rb):
		team1 = self.red if rb else self.blue
		team2 = self.blue if rb else self.red

		for pr in team1:
			if pr.wetness >= 100:continue
			if pr.cooldown != 0:
				continue
			if pr.thx != -1: continue
			idx = -1
			maxsh = -20000000
			for i, pb in enumerate(team2):
				if pb.wetness >= 100:continue
				dsh = pr.coord.manhattan_to(pb.coord)
				if dsh <= pr.optimalRange:
					if pb.wetness > maxsh:
						maxsh = pb.wetness
						idx = i

			if idx != -1:
				cover = self.get_cover_modifier(pr, team2[idx])
				team2[idx].wetness += pr.soakingPower * cover
				##print(f"{'Red' if rb else 'Blue'} agent at {pr.coord} shoots at {team2[idx].coord} -> wetness: {team2[idx].wetness}")
				if team2[idx].wetness >= 100:
					team2[idx].dead = 1.0

			pr.cooldown = pr.mx_cooldown

	

	def get_FloorScore(self):

		my_count = 0
		opp_count = 0

		for y in range(self.grid.height):
			for x in range(self.grid.width):
				best_dist1 = float('inf')
				best_dist2 = float('inf')

				for a in self.red:
					da = abs(y - a.coord.y) + abs(x - a.coord.x)
					if a.wetness >= 50:
						da *= 2
					best_dist1 = min(best_dist1, da)

				for a in self.blue:
					da = abs(y - a.coord.y) + abs(x - a.coord.x)
					if a.wetness >= 50:
						da *= 2
					best_dist2 = min(best_dist2, da)

				if best_dist1 < best_dist2:
					my_count += 1
				elif best_dist2 < best_dist1:
					opp_count += 1

		r = my_count - opp_count
		if r > 0:
			self.rscore += r
		else:
			self.bscore += -r

		return r


	def get_cover_modifier(self, target, shooter):
		dx = target.coord.x - shooter.coord.x
		dy = target.coord.y - shooter.coord.y
		best_modifier = 1.0

		for d in [(dx, 0), (0, dy)]:
			if abs(d[0]) > 1 or abs(d[1]) > 1:
				adj_x = -int(math.copysign(1, d[0])) if d[0] != 0 else 0
				adj_y = -int(math.copysign(1, d[1])) if d[1] != 0 else 0

				cover_pos = Coord(target.coord.x + adj_x, target.coord.y + adj_y)

				if cover_pos.chebyshev_to(shooter.coord) > 1:
					tile = self.grid.get(cover_pos.x, cover_pos.y)
					best_modifier = min(best_modifier, tile.get_cover_modifier())

		return best_modifier




def spawn(grid):
	left_players = []   # Equipe rouge
	right_players = []  # Equipe bleue

	for c in grid.spawns:
		left_players.append(Player(c, "red"))
		right_players.append(Player(grid.opposite(c), "blue"))

	return left_players, right_players


#------------------END GAME---------------------------------
#-------------------ENCODING NN ----------------------------
import torch

def encode_playersTALLNumpy2(indp, players, players2, grid, grid_height, grid_width, game):
	tensor = np.zeros((11, grid_height, grid_width), dtype=np.float32)

	player_a = players + players2
	limit = len(players)

	flip = (indp >= limit)  # True si joueur est dans l’équipe bleue

	index = 0
	for player in player_a:
		x, y = player.coord.x, player.coord.y

		if flip:
			x = grid_width - 1 - x  # miroir horizontal

		if 0 <= x < grid_width and 0 <= y < grid_height:
			tensor[0, y, x] = player.cooldown / player.mx_cooldown
			tensor[1, y, x] = player.splash_bombs / 3.0
			tensor[2, y, x] = player.wetness / 100.0
			tensor[3, y, x] = (player.optimalRange - 5) / 5.0
			tensor[4, y, x] = (player.soakingPower - 10) / 15.0

			if index == indp:
				tensor[5, y, x] = 1.0  # joueur courant

			if (indp < limit and index < limit) or (indp >= limit and index >= limit):
				tensor[6, y, x] = 1.0  # allié
			else:
				tensor[7, y, x] = 1.0  # ennemi

			tensor[8, y, x] = player.dead
			score = game.rscore if player.team == 'red' else game.bscore
			tensor[9, y, x] = score / 1500.0

		index += 1

	# ⚡ miroir du décor
	for y in range(grid_height):
		for x in range(grid_width):
			xm = grid_width - 1 - x if flip else x
			cell = grid.get(x, y)
			t = cell.get_type()
			if t == Tile.TYPE_FLOOR:
				tensor[10, y, xm] = 0.25
			elif t == Tile.TYPE_LOW_COVER:
				tensor[10, y, xm] = 0.75
			elif t == Tile.TYPE_HIGH_COVER:
				tensor[10, y, xm] = 1.0

	return tensor


def encode_playersTALLNumpy(indp, players, players2, grid, grid_height, grid_width, game):

	tensor = np.zeros((11, grid_height, grid_width), dtype=np.float32)

	player_a = []
	
	player_a.extend(players)
	player_a.extend(players2)

	limit = len(players)

	index = 0
	base = 0
	for player in player_a:
		x, y = player.coord.x, player.coord.y

		
		# �vite les d�bordements hors grille
		if  (0 <= x < grid_width and 0 <= y < grid_height):
			
			tensor[0, y, x] = player.cooldown / player.mx_cooldown  # cooldown norm.
			tensor[1, y, x] = player.splash_bombs / 3.0              # max bombs = 3
			tensor[2, y, x] = player.wetness / 100.0                 # si born� � 100 ?
			tensor[3, y, x] = (player.optimalRange - 5) / 5.0        # de 5 � 10
			tensor[4, y, x] = (player.soakingPower - 10) / 15.0      # de 10 � 25

			if index == indp:
				tensor[5, y, x] = 1.0  # canal red

			if (indp < limit and index < limit) or (indp >= limit and index >= limit):
				tensor[6, y, x] = 1.0  

			if (indp < limit and index >= limit) or (indp >= limit and index < limit):
				tensor[7, y, x] = 1.0  # ennemi
	
			tensor[8, y, x] = player.dead
			score = game.rscore if player.team == 'red' else game.bscore
			tensor[9, y, x] = score / 1500.0
			
		index += 1

	#print("indp=", indp, "ally mask=", tensor[6].sum(), "enemy mask=", tensor[7].sum())

	
	for y in range(10):
		for x in range(20):
			cell = grid.get(x, y)
			t = cell.get_type()
			if t == Tile.TYPE_FLOOR:
				tensor[10, y, x] = 0.25
			elif t == Tile.TYPE_LOW_COVER:
				tensor[10, y, x] = 0.75
			elif t == Tile.TYPE_HIGH_COVER:
				tensor[10, y, x] = 1.0

	return tensor  


def encode_playersTALL(indp, players, players2, grid, grid_height, grid_width, game):

	tensor = torch.zeros((11, grid_height, grid_width), dtype=torch.float32)

	player_a = []
	
	player_a.extend(players)
	player_a.extend(players2)
	limit = len(players)
	
	index = 0
	base = 0
	for player in player_a:
		x, y = player.coord.x, player.coord.y

		# �vite les d�bordements hors grille
		if  (0 <= x < grid_width and 0 <= y < grid_height):
			
			tensor[0, y, x] = player.cooldown / player.mx_cooldown  # cooldown norm.
			tensor[1, y, x] = player.splash_bombs / 3.0              # max bombs = 3
			tensor[2, y, x] = player.wetness / 100.0                 # si born� � 100 ?
			tensor[3, y, x] = (player.optimalRange - 5) / 5.0        # de 5 � 10
			tensor[4, y, x] = (player.soakingPower - 10) / 15.0      # de 10 � 25

			if index == indp:
				tensor[5, y, x] = 1.0  # canal red
	
			if (indp < limit and index < limit) or (indp >= limit and index >= limit):
				tensor[6, y, x] = 1.0  

			if (indp < limit and index >= limit) or (indp >= limit and index < limit):
				tensor[7, y, x] = 1.0  # ennemi

			tensor[8, y, x] = player.dead
			score = game.rscore if player.team == 'red' else game.bscore
			tensor[9, y, x] = score / 1500.0

		index += 1

	
	for y in range(10):
		for x in range(20):
			cell = grid.get(x, y)
			t = cell.get_type()
			if t == Tile.TYPE_FLOOR:
				tensor[10, y, x] = 0.25
			elif t == Tile.TYPE_LOW_COVER:
				tensor[10, y, x] = 0.75
			elif t == Tile.TYPE_HIGH_COVER:
				tensor[10, y, x] = 1.0

	return tensor  


def encode_playersT(players, grid_height, grid_width, game):
	# On utilise 7 canaux :
	# cooldown, bombs, wetness, range, power, is_red, is_blue
	tensor = torch.zeros((45, grid_height, grid_width), dtype=torch.float32)

	player_a = []
	index = 0
	if len(players) == 0:
		return tensor

	while 1:
		for p in players:
			player_a.append(p)
			index += 1
			if index == 5:break
		if index == 5:break

	base = 0
	for player in player_a:
		x, y = player.coord.x, player.coord.y

		# �vite les d�bordements hors grille
		if  (0 <= x < grid_width and 0 <= y < grid_height):
			
			tensor[base+0, y, x] = player.cooldown / player.mx_cooldown  # cooldown norm.
			tensor[base+1, y, x] = player.splash_bombs / 3.0              # max bombs = 3
			tensor[base+2, y, x] = player.wetness / 100.0                 # si born� � 100 ?
			tensor[base+3, y, x] = (player.optimalRange - 5) / 5.0        # de 5 � 10
			tensor[base+4, y, x] = (player.soakingPower - 10) / 15.0      # de 10 � 25

			if player.team == "red":
				tensor[base+5, y, x] = 1.0  # canal red
			elif player.team == "blue":
				tensor[base+6, y, x] = 1.0  # canal blue

			tensor[base+7, y, x] = player.dead
			score = game.rscore if players[0].team == 'red' else game.bscore
			tensor[base + 8, y, x] = score / 1500.0

			base += 9

	return tensor  # shape : (7, H, W)

def encode_players(players, grid_height, grid_width, game):
	# On utilise 7 canaux :
	# cooldown, bombs, wetness, range, power, is_red, is_blue
	tensor = torch.zeros((45, grid_height, grid_width), dtype=torch.float32)

	base = 0
	for player in players:
		x, y = player.coord.x, player.coord.y

		# �vite les d�bordements hors grille
		if  (0 <= x < grid_width and 0 <= y < grid_height):
			
			tensor[base+0, y, x] = player.cooldown / player.mx_cooldown  # cooldown norm.
			tensor[base+1, y, x] = player.splash_bombs / 3.0              # max bombs = 3
			tensor[base+2, y, x] = player.wetness / 100.0                 # si born� � 100 ?
			tensor[base+3, y, x] = (player.optimalRange - 5) / 5.0        # de 5 � 10
			tensor[base+4, y, x] = (player.soakingPower - 10) / 15.0      # de 10 � 25

			if player.team == "red":
				tensor[base+5, y, x] = 1.0  # canal red
			elif player.team == "blue":
				tensor[base+6, y, x] = 1.0  # canal blue

			tensor[base+7, y, x] = player.dead
			score = game.rscore if players[0].team == 'red' else game.bscore
			tensor[base + 8, y, x] = score / 1500.0

			base += 9

	return tensor  # shape : (7, H, W)


def encode_grid(grid):

	tensor = torch.zeros((3, 10, 20), dtype=torch.float32)

	w, h = 20, 10
	for y in range(h):
		for x in range(w):
			cell = grid.get(x, y)
			t = cell.get_type()
			if t == Tile.TYPE_FLOOR:
				tensor[0, y, x] = 1.0
			elif t == Tile.TYPE_LOW_COVER:
				tensor[1, y, x] = 1.0
			elif t == Tile.TYPE_HIGH_COVER:
				tensor[2, y, x] = 1.0

	return tensor

import numpy as np

def encode_players_numpy(players, grid_height, grid_width, game):
	# On utilise 8 canaux par joueur (comme dans ton code PyTorch)
	# cooldown, bombs, wetness, range, power, is_red, is_blue, dead
	tensor = np.zeros((45, grid_height, grid_width), dtype=np.float32)

	player_a = players
	#index = 0
	#if len(players) == 0:
	#	return tensor

	#while 1:
	#	for p in players:
	#		player_a.append(p)
	#		index += 1
	#		if index == 5:break
	#	if index == 5:break

	base = 0
	for player in player_a:
		x, y = player.coord.x, player.coord.y

		# évite les débordements hors grille
		if 0 <= x < grid_width and 0 <= y < grid_height:
			tensor[base + 0, y, x] = player.cooldown / player.mx_cooldown
			tensor[base + 1, y, x] = player.splash_bombs / 3.0
			tensor[base + 2, y, x] = player.wetness / 100.0
			tensor[base + 3, y, x] = (player.optimalRange - 5) / 5.0
			tensor[base + 4, y, x] = (player.soakingPower - 10) / 15.0

			if player.team == "red":
				tensor[base + 5, y, x] = 1.0
			elif player.team == "blue":
				tensor[base + 6, y, x] = 1.0

			tensor[base + 7, y, x] = player.dead
			score = game.rscore if players[0].team == 'red' else game.bscore
			tensor[base + 8, y, x] = score / 1500.0

			base += 9

	return tensor  # shape : (40, H, W)


def encode_grid_numpy(grid):
	tensor = np.zeros((3, 10, 20), dtype=np.float32)

	w, h = 20, 10
	for y in range(h):
		for x in range(w):
			cell = grid.get(x, y)
			t = cell.get_type()
			if t == Tile.TYPE_FLOOR:
				tensor[0, y, x] = 1.0
			elif t == Tile.TYPE_LOW_COVER:
				tensor[1, y, x] = 1.0
			elif t == Tile.TYPE_HIGH_COVER:
				tensor[2, y, x] = 1.0

	return tensor  # shape : (3, 10, 20)


def create_dead_player(coord, team):
	p = Player(coord, team)
	p.cooldown = 0
	p.mx_cooldown = 1
	p.splash_bombs = 0
	p.wetness = 0
	p.optimalRange = 0
	p.soakingPower = 0
	p.score = 0
	p.dead = 1
	return p


def complete_team(players, team, n=5):
	# Garde les joueurs vivants
	players_completed = players.copy()
	
	# Calcule combien il manque de joueurs
	missing = n - len(players)
	
	# Ajoute les joueurs morts manquants
	if missing > 0:
		dead_players = [
			create_dead_player(Coord(-1, -1), team)
			for _ in range(missing)
		]
		players_completed.extend(dead_players)
	
	return players_completed


def encode_ALL_RL(indp, grid, red, blue, game):
	#red_complete = complete_team(red, "red", 5)
	#blue_complete = complete_team(blue, "blue", 5)

	#tensor_red = encode_playersT(red, 10, 20, game)
	#tensor_blue = encode_playersT(blue, 10, 20, game)
	#tensor_grid = encode_grid(grid)

	##print(tensor_red.shape)   # (channels_red, H, W)
	##print(tensor_blue.shape)  # (channels_blue, H, W)
	##print(tensor_grid.shape)  # (channels_grid, H, W)

	#input_tensor = torch.cat([
	#	tensor_red,
	#	tensor_blue, 
	#	tensor_grid,
	#], dim=0)

	input_tensor = encode_playersTALL(indp, red, blue, grid, 10, 20, game)



	return input_tensor

import numpy as np

def encode_ALL_RL_numpy(indp, grid, red, blue, game):
	#red_complete = complete_team(red, "red", 5)
	#blue_complete = complete_team(blue, "blue", 5)

	#tensor_red = encode_players_numpy(red, 10, 20, game)   # (40, 20, 10)
	#tensor_blue = encode_players_numpy(blue, 10, 20, game) # (40, 20, 10)
	#tensor_grid = encode_grid_numpy(grid)                     # (3, 20, 10)

	# concaténation sur l'axe des canaux (axis=0)
	#input_tensor = np.concatenate([tensor_red, tensor_blue, tensor_grid], axis=0) 


	input_tensor = encode_playersTALLNumpy(indp, red, blue, grid, 10, 20, game)

	return input_tensor  # shape: (40+40+3=83, 20, 10)


import torch.nn as nn
import torch.nn.functional as F

class PolicyNetB(nn.Module):
	def __init__(self, num_players=10, num_actions=5):
		super().__init__()
		self.num_players = num_players
		self.num_actions = num_actions

		self.conv1 = nn.Conv2d(83, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

		self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # sort du (B, C, 1, 1)
		self.fc = nn.Linear(64, num_players * num_actions)

	def forward(self, x):
		x = F.relu(self.conv1(x))  # (B, 32, H, W)
		x = F.relu(self.conv2(x))  # (B, 64, H, W)
		x = F.relu(self.conv3(x))  # (B, 64, H, W)

		x = self.global_pool(x)  # (B, 64, 1, 1)
		x = x.view(x.size(0), -1)  # (B, 64)

		out = self.fc(x)  # (B, num_players * num_actions)
		out = out.view(-1, self.num_players, self.num_actions)  # (B, 4, 5)
		return out


class ValueNetB(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(83, 64, kernel_size=3, padding=1)   # (batch, 83, 20, 10) -> (batch, 64, 20, 10)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # -> (batch, 128, 20, 10)
		self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # -> (batch, 256, 20, 10)

		self.pool = nn.AdaptiveAvgPool2d((1, 1))  # r�duit chaque canal � une seule valeur
		self.fc = nn.Linear(256, 10)  # une seule valeur : V(s)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = self.pool(x)             # (batch, 256, 1, 1)
		x = x.view(x.size(0), -1)    # (batch, 256)
		value = self.fc(x)           # (batch, 1)
		return value

class PolicyNetA(nn.Module):
	def __init__(self, num_players=10, num_actions=5):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(83, 8, 3, padding=1), nn.ReLU(),
			nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
			nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()
		)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(16, num_players * num_actions)
		self.num_players = num_players
		self.num_actions = num_actions

	def forward(self, x):
		x = self.pool(self.conv(x)).view(x.size(0), -1)
		return self.fc(x).view(-1, self.num_players, self.num_actions)

class ValueNetA(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(83, 8, 3, padding=1), nn.ReLU(),
			nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
			nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()
		)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(32, 10)

	def forward(self, x):
		x = self.pool(self.conv(x)).view(x.size(0), -1)
		return self.fc(x)

class PolicyNetOO(nn.Module):
	def __init__(self, num_players=5, num_actions=5):
		super().__init__()
		self.conv1 = nn.Conv2d(93, 8, 3, padding=1)
		self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
		self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
		self.relu = nn.ReLU()
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Linear(16, 64)
		self.fc2 = nn.Linear(64, 128)
		self.fc3 = nn.Linear(128, num_players * num_actions)
		
		self.dropout1 = nn.Dropout(p=0.3)  # Dropout après fc1
		self.dropout2 = nn.Dropout(p=0.3)  # Dropout après fc2 (optionnel)

		self.num_players = num_players
		self.num_actions = num_actions

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.pool(x).view(x.size(0), -1)
		x = self.relu(self.fc1(x))
		x = self.dropout1(x)
		x = self.relu(self.fc2(x))
		x = self.dropout2(x)
		return self.fc3(x).view(-1, self.num_players, self.num_actions)

import torch
import torch.nn as nn

class PolicyNetBC(nn.Module):
	def __init__(self, num_players=5, num_actions=5):
		super().__init__()
		self.conv1 = nn.Conv2d(93, 8, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(8)

		self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(16)

		self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
		self.bn3 = nn.BatchNorm2d(16)

		self.pool = nn.AdaptiveAvgPool2d(1)

		self.fc1 = nn.Linear(16, 64)
		self.bn_fc1 = nn.BatchNorm1d(64)

		self.fc2 = nn.Linear(64, 128)
		self.bn_fc2 = nn.BatchNorm1d(128)

		self.fc3 = nn.Linear(128, num_players * num_actions)

		self.dropout1 = nn.Dropout(p=0.3)
		self.dropout2 = nn.Dropout(p=0.3)

		self.relu = nn.ReLU()

		self.num_players = num_players
		self.num_actions = num_actions

	def forward(self, x):
		x = self.relu(self.bn1(self.conv1(x)))
		x = self.relu(self.bn2(self.conv2(x)))
		x = self.relu(self.bn3(self.conv3(x)))
		x = self.pool(x).view(x.size(0), -1)  # shape: (B, 16)
		
		x = self.relu(self.bn_fc1(self.fc1(x)))
		x = self.dropout1(x)
		x = self.relu(self.bn_fc2(self.fc2(x)))
		x = self.dropout2(x)

		return self.fc3(x).view(-1, self.num_players, self.num_actions)



class ValueNetBC(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(93, 16, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(16)

		self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(32)

		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.bn3 = nn.BatchNorm2d(64)

		self.pool = nn.AdaptiveAvgPool2d(1)

		self.fc1 = nn.Linear(64, 64)
		self.bn_fc1 = nn.BatchNorm1d(64)

		self.dropout = nn.Dropout(p=0.3)
		self.fc2 = nn.Linear(64, 5)  # ou 1 si score global

		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.bn1(self.conv1(x)))
		x = self.relu(self.bn2(self.conv2(x)))
		x = self.relu(self.bn3(self.conv3(x)))
		x = self.pool(x).view(x.size(0), -1)

		x = self.relu(self.bn_fc1(self.fc1(x)))
		x = self.dropout(x)

		return self.fc2(x)


class PolicyNet(nn.Module):
	def __init__(self, num_players=5, num_actions=5):
		super().__init__()
		self.conv1 = nn.Conv2d(93, 8, 3, padding=1)
		
		self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
		
		self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
		
		self.pool = nn.AdaptiveAvgPool2d(1)

		self.fc1 = nn.Linear(16, 64)
		
		self.fc2 = nn.Linear(64, 128)

		self.fc3 = nn.Linear(128, num_players * num_actions)

		#self.dropout1 = nn.Dropout(p=0.3)
		#self.dropout2 = nn.Dropout(p=0.3)

		#self.relu = nn.ReLU()
		self.relu = nn.Tanh()

		self.num_players = num_players
		self.num_actions = num_actions

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.pool(x).view(x.size(0), -1)  # shape: (B, 16)
		
		x = self.relu(self.fc1(x))
		#x = self.dropout1(x)
		x = self.relu(self.fc2(x))
		#x = self.dropout2(x)

		return self.fc3(x).view(-1, self.num_players, self.num_actions)



class ValueNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(93, 16, 3, padding=1)
		
		self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
	
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
	
		self.pool = nn.AdaptiveAvgPool2d(1)

		self.fc1 = nn.Linear(64, 64)
	
		#self.dropout = nn.Dropout(p=0.3)
		self.fc2 = nn.Linear(64, 5)  # ou 1 si score global

		#self.relu = nn.ReLU()
		self.relu = nn.Tanh()

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.pool(x).view(x.size(0), -1)

		x = self.relu(self.fc1(x))
		
		return self.fc2(x)


# ===== DQN discret pour états (C,H,W) et 5 actions =====
# - Réseau conv → Q(s,a) pour chaque action
# - ε-greedy
# - Experience Replay + Target Network
# - Huber loss
# - Soft update (Polyak) optionnel

import math, random, collections, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------
# Hyperparamètres (ajuste si besoin)
# -------------------------------------------------------
NUM_ACTIONS   = 5      # ex: [haut, bas, gauche, droite, rien]
STATE_CHANNELS= 11     # ex: 93
HEIGHT, WIDTH = 10, 20
BUFFER_SIZE   = 100_000
BATCH_SIZE    = 64
GAMMA         = 0.99
LR            = 2.5e-4
TARGET_UPDATE = 1000     # tous les N steps on copie vers le target (hard update)
TAU           = 0.005       # si >0, on fait un soft update (Polyak). Laisse 0 si tu utilises TARGET_UPDATE
EPS_START     = 1.0
EPS_END       = 0.1
EPS_DECAY     = 15000     # plus c'est grand, plus epsilon décroît lentement
GRAD_NORM_CLIP= 1.0
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# Réseau Q conv: entrée (B,C,H,W) → sortie (B, NUM_ACTIONS)
# -------------------------------------------------------
class ConvDQN(nn.Module):
	def __init__(self, in_channels=STATE_CHANNELS, num_actions=NUM_ACTIONS):
		super().__init__()
		# Petit réseau conv (adapte si besoin)
		self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
		self.pool  = nn.AdaptiveAvgPool2d(1)   # -> (B,64,1,1)
		self.fc1   = nn.Linear(16, 64)
		self.fc2   = nn.Linear(64, num_actions)

		nn.init.uniform_(self.fc2.weight, -0.1, 0.1)
		nn.init.constant_(self.fc2.bias, 0.0)

	def forward(self, x):
		# x: (B,C,H,W)
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = self.pool(x).view(x.size(0), -1)  # (B,64)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)                    # (B, NUM_ACTIONS) = Q(s,·)
		return x

# -------------------------------------------------------
# Replay Buffer
# -------------------------------------------------------
Transition = collections.namedtuple(
	"Transition", ("state", "action", "reward", "next_state", "done")
)

class ReplayBuffer:
	def __init__(self, capacity=BUFFER_SIZE):
		self.buf = collections.deque(maxlen=capacity)

	def push(self, state, action, reward, next_state, done):
		# state / next_state: np.array ou torch.Tensor (C,H,W)
		# action: int
		# reward: float
		# done: bool (ou 0/1)
		self.buf.append(Transition(state, action, reward, next_state, done))

	def sample(self, batch_size=BATCH_SIZE):
		batch = random.sample(self.buf, batch_size)
		return Transition(*zip(*batch))

	def __len__(self):
		return len(self.buf)

# -------------------------------------------------------
# Agent DQN
# -------------------------------------------------------
class DQNAgent:
	def __init__(self, state_channels=STATE_CHANNELS, num_actions=NUM_ACTIONS):
		self.num_actions = num_actions
		self.q_net       = ConvDQN(state_channels, num_actions).to(DEVICE)
		self.q_target    = ConvDQN(state_channels, num_actions).to(DEVICE)
		self.q_target.load_state_dict(self.q_net.state_dict())
		self.optimizer   = torch.optim.Adam(self.q_net.parameters(), lr=LR)
		self.replay      = ReplayBuffer()
		self.train_steps = 0
		self.eps         = EPS_START

	@torch.no_grad()
	def select_action22(self, state):
		"""
		state: torch.Tensor (C,H,W) ou np.ndarray (C,H,W)
		return: int, action choisie pour le joueur
		"""
		if isinstance(state, np.ndarray):
			state = torch.from_numpy(state).float()
		state = state.unsqueeze(0).to(DEVICE)  # (1,C,H,W)

		# ε-greedy
		self.eps = EPS_END + (EPS_START - EPS_END) * math.exp(
			-1.0 * self.train_steps / EPS_DECAY
		)

		q_values = self.q_net(state)  # (1, num_actions)
		q_values = q_values[0]        # (num_actions,)

		if random.random() < self.eps:
			action = random.randint(0, 4)
		else:
			action = q_values.argmax(dim=0).item()

		return action

	@torch.no_grad()
	def select_action(self, state, valid_actions=None):
		"""
		state: torch.Tensor (C,H,W)
		valid_actions: liste/bool mask de taille 5 (True=valide, False=interdit)
		return: int (0..4)
		"""
		if state.ndim == 3:  # (C,H,W)
			state = state.unsqueeze(0).to(DEVICE)  # (1,C,H,W)

		q_values = self.q_net(state)[0]  # (5,) logits ou Q-values

		if valid_actions is not None:
			mask = torch.tensor(valid_actions, dtype=torch.bool, device=q_values.device)
			# mettre -inf aux actions invalides
			q_values[~mask] = -1e9

		self.eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * self.train_steps / EPS_DECAY)
		# epsilon-greedy
		if random.random() < self.eps:
			if valid_actions is None:
				action = random.randint(0, 4)
			else:
				valid_indices = [i for i, ok in enumerate(valid_actions) if ok]
				action = random.choice(valid_indices)
		else:
			action = q_values.argmax().item()

		return action


	def optimize(self):
		if len(self.replay) < BATCH_SIZE:
			return None

		# --- Sample du replay buffer
		state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay.sample(BATCH_SIZE)

		state_batch      = torch.tensor(np.array(state_batch), dtype=torch.float32, device=DEVICE)      # (B,C,H,W)
		next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32, device=DEVICE) # (B,C,H,W)
		action_batch     = torch.tensor(np.array(action_batch), dtype=torch.long, device=DEVICE)        # (B,)
		reward_batch     = torch.tensor(np.array(reward_batch), dtype=torch.float32, device=DEVICE)     # (B,)
		done_batch       = torch.tensor(np.array(done_batch), dtype=torch.float32, device=DEVICE)       # (B,)

		norm_reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-8)

		# --- Q(s,a) courant
		q_values = self.q_net(state_batch)              # (B, NUM_ACTIONS)
		q_sa = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)  # (B,)

		with torch.no_grad():
			# --- Double DQN
			next_q_values = self.q_net(next_state_batch)             # (B, NUM_ACTIONS)
			next_actions = next_q_values.argmax(dim=1, keepdim=True) # (B,1)

			next_q_target = self.q_target(next_state_batch)          # (B, NUM_ACTIONS)
			next_q = next_q_target.gather(1, next_actions).squeeze(1)  # (B,)

			# --- Cible
			target = norm_reward_batch + (1.0 - done_batch) * GAMMA * next_q
			#target = 0.9 * target + 0.1 * q_sa.detach()  # smoothing (optionnel)

		# --- Loss
		loss = F.smooth_l1_loss(q_sa, target)

		self.optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
		self.optimizer.step()
		self.train_steps += 1

		# --- Update du target net
		if TAU > 0.0:  # soft update
			with torch.no_grad():
				for p, tp in zip(self.q_net.parameters(), self.q_target.parameters()):
					tp.data.mul_(1.0 - TAU).add_(TAU * p.data)
		elif self.train_steps % TARGET_UPDATE == 0:  # hard update
			self.q_target.load_state_dict(self.q_net.state_dict())

		return loss.item()


# -------------------------------------------------------
# Exemple d’intégration (pseudo-boucle d’entraînement)
# -------------------------------------------------------
"""
agent = DQNAgent(state_channels=93, num_actions=5)

for episode in range(1, 5001):
	state = encode_ALL_RL(env_grid, players_myteam, players_enemy, game)  # -> torch.Tensor ou np.ndarray (C,H,W)
	done = False
	ep_return = 0.0

	while not done:
		action = agent.select_action(state)  # int in [0..4]
		# -> exécuter l’action dans ton jeu
		# next_state, reward, done, info = env.step(action)
		# Ici, adapte à ton moteur: reward = float, done = bool, next_state (C,H,W)

		agent.replay.push(state, action, reward, next_state, float(done))
		loss = agent.optimize()

		state = next_state
		ep_return += reward

	print(f"Episode {episode} | Return {ep_return:.2f} | eps={agent.eps:.3f} | loss={loss}")
"""

# -------------------------------------------------------
# Aide: mapping action index -> mouvement
# -------------------------------------------------------
"""
# indices 0..4 -> haut, bas, gauche, droite, rien
DIRS = [(0,-1),(0,1),(-1,0),(1,0),(0,0)]
dx, dy = DIRS[action]
"""



import torch

def compute_reward_from_playersDR(game, players: list[Player], players2: list[Player], score_territory):
	N = len(players)

	# R�cup�rer les features
	wetness = torch.tensor([p.wetness for p in players], dtype=torch.float)
	dead = torch.tensor([p.dead for p in players], dtype=torch.float)
	bombs = torch.tensor([p.splash_bombs for p in players], dtype=torch.float)
	optimal_range = torch.tensor([p.optimalRange for p in players], dtype=torch.float)
	soaking_power = torch.tensor([p.soakingPower for p in players], dtype=torch.float)

	# Ici on utilise score_territory global pour tous
	scores = torch.full((N,), score_territory, dtype=torch.float)

	# Normalisation min-max
	def minmax_norm(x, min_val, max_val):
		x_clamped = torch.clamp(x, min_val, max_val)
		return (x_clamped - min_val) / (max_val - min_val + 1e-8)

	scores_norm = minmax_norm(scores, 0, 200)
	bombs_norm = minmax_norm(bombs, 0, 5)
	wetness_norm = minmax_norm(wetness, 0, 100)
	dead_norm = minmax_norm(dead, 0, 1)
	soaking_power_norm = minmax_norm(soaking_power, 10, 30)

	reward = scores_norm + bombs_norm * 2 - wetness_norm * 0.5 - dead_norm * 1.0

	reward = torch.zeros(N, dtype=torch.float)

	positions = torch.tensor([[p.coord.x, p.coord.y] for p in players], dtype=torch.float)
	positions2 = torch.tensor([[p.coord.x, p.coord.y] for p in players2], dtype=torch.float)

	for i in range(N):
		my_pos = positions[i]
		r = optimal_range[i]
		sp = soaking_power_norm[i]

		for j in range(len(players2)):
			if not players2[j].dead:
				enemy_pos = positions2[j]
				d = torch.norm(my_pos - enemy_pos, p=2)  # euclidean, ou utilise p=1 pour manhattan si tu préfères

				if abs(d.item() - r.item()) <= 2.0:
					reward[i] += sp * 0.3


	reward_min = reward.min()
	reward_max = reward.max()
	reward_norm = (reward - reward_min) / (reward_max - reward_min + 1e-8)

	count = 5
	for idx, p in enumerate(players):
		if p.wetness >= 100 or p.dead == 1.0:
			reward_norm[idx] = 0.0
			count-=1

	return reward_norm


def compute_distance_map(H, W, units):
	"""Retourne une grille (H,W) avec la distance minimale de chaque case à un joueur."""
	grid = np.full((H, W), np.inf)

	for unit in units:
		uy, ux = unit.coord.y, unit.coord.x
		wet_factor = 2 if unit.wetness >= 50 else 1
		dists = np.abs(np.arange(H)[:, None] - uy) + np.abs(np.arange(W)[None, :] - ux)
		grid = np.minimum(grid, dists * wet_factor)
	return grid

def find_best_spot_numpy_general(game, color: str):
	H, W = game.grid.height, game.grid.width
	red_players = game.red
	blue_players = game.blue

	# 1. distance to red / blue
	dist_red = compute_distance_map(H, W, red_players)
	dist_blue = compute_distance_map(H, W, blue_players)

	# 2. current control map
	control_map = np.zeros((H, W), dtype=np.int8)
	control_map[dist_red < dist_blue] = 1   # red controls
	control_map[dist_red > dist_blue] = -1  # blue controls

	base_score = np.sum(control_map)

	# Determine which player to simulate
	simulate_as_red = color.lower() == "red"
	dist_me = dist_red if simulate_as_red else dist_blue
	dist_opp = dist_blue if simulate_as_red else dist_red
	current_sign = 1 if simulate_as_red else -1

	best_gain = float('-inf')
	best_pos = (0, 0)

	for y in range(H):
		for x in range(W):
			# Simulate new distance map by adding a player at (x, y)
			d = np.abs(np.arange(H)[:, None] - y) + np.abs(np.arange(W)[None, :] - x)
			new_dist_me = np.minimum(dist_me, d)

			# Recalculate new control map
			new_control = np.zeros((H, W), dtype=np.int8)
			new_control[new_dist_me < dist_opp] = current_sign
			new_control[new_dist_me > dist_opp] = -current_sign
			new_score = np.sum(new_control)

			gain = new_score - base_score
			if gain > best_gain:
				best_gain = gain
				best_pos = (x, y)

	return best_pos, best_gain


def compute_reward_from_playersE(game, players: list[Player], players2: list[Player], score_territory):
	N = len(players)

	# Score global identique pour tous
	scores = torch.full((N,), score_territory, dtype=torch.float)

	# Positions et états
	pos1 = torch.tensor([[p.coord.x, p.coord.y] for p in players], dtype=torch.float)
	pos2 = torch.tensor([[p.coord.x, p.coord.y] for p in players2], dtype=torch.float)
	dead2 = torch.tensor([p.dead for p in players2], dtype=torch.bool)

	# Distances entre chaque joueur de players et chaque joueur de players2
	dists = torch.cdist(pos1, pos2, p=2)

	# Remplir de +inf les ennemis morts
	dists[:, dead2] = float('inf')

	# Distance minimale à un ennemi vivant pour chaque joueur
	closest_enemy_dists = torch.min(dists, dim=1).values

	# Inverser les distances : plus proche = meilleur
	inv_dists = 1.0 / (closest_enemy_dists + 1e-6)

	# Normalisation du score territoire
	scores_norm = (scores - 0) / (200 - 0 + 1e-8)

	# Combinaison score + proximité ennemie
	reward = scores_norm + inv_dists * 0.1  # facteur ajustable

	# Normalisation finale
	reward_min = reward.min()
	reward_max = reward.max()
	reward_norm = (reward - reward_min) / (reward_max - reward_min + 1e-8)

	return reward_norm

def compute_reward_from_playersAAA(game, players: list[Player], players2: list[Player], score_territory):
	
	N = len(players)

	# Score global de territoire actuel
	scores = torch.full((N,), score_territory, dtype=torch.float)

	# Positions des joueurs
	pos1 = torch.tensor([[p.coord.x, p.coord.y] for p in players], dtype=torch.float)
	pos2 = torch.tensor([[p.coord.x, p.coord.y] for p in players2], dtype=torch.float)
	dead2 = torch.tensor([p.dead for p in players2], dtype=torch.bool)

	# Distance la plus proche à un ennemi vivant
	dists = torch.cdist(pos1, pos2, p=2)
	dists[:, dead2] = float('inf')
	closest_enemy_dists = torch.min(dists, dim=1).values
	inv_dists = 1.0 / (closest_enemy_dists + 1e-6)

	# Normalisation du score territoire actuel
	scores_norm = (scores - 0) / (2000 - 0 + 1e-8)

	# Meilleure case stratégique à atteindre
	team_color = players[0].team
	best_pos, territory_gain = find_best_spot_numpy_general(game, team_color)
	best_pos_tensor = torch.tensor(best_pos, dtype=torch.float)

	# Distance à la meilleure case
	# pos1: (N, 2)  - positions à évaluer
	# best_pos_tensor: (M, 2) - positions de référence

	# On étend les dimensions pour calculer toutes les distances pairwise
	# pos1[:, None, :] => (N, 1, 2)
	# best_pos_tensor[None, :, :] => (1, M, 2)
	#print("pos1 shape:", pos1.shape)
	#print("best_pos_tensor shape:", best_pos_tensor.shape)

	if best_pos_tensor.dim() == 1:
		best_pos_tensor = best_pos_tensor.unsqueeze(0)  # passe de (2,) à (1, 2)

	dists = torch.norm(pos1[:, None, :] - best_pos_tensor[None, :, :], dim=2)  # (N, M)

	# Pour chaque pos1[i], on garde seulement la distance minimale
	min_dist_to_best = torch.min(dists, dim=1).values  # (N,)

	# Optionnel : score de proximité (inverse)
	proximity_to_closest = 1.0 / (min_dist_to_best + 1e-6)


	# Gain potentiel de territoire sur best_pos
	#territory_gain = torch.tensor([territory_gain], dtype=torch.float).expand(N)

	# 🧮 Reward : territoire potentiel + proximité best_pos + score + pression ennemie
	reward = (
		#territory_gain * 0.0005 +
		proximity_to_closest * 0.7 +
		scores_norm * 0.1 +
		inv_dists * 0.2
	)

	# Normalisation finale
	reward_min = reward.min()
	reward_max = reward.max()
	reward_norm = (reward - reward_min) / (reward_max - reward_min + 1e-8)

	count = 5
	for idx, p in enumerate(players):
		if p.wetness >= 100 or p.dead == 1.0:
			reward_norm[idx] = 0.0
			count-=1
	#print(reward_norm)

	return reward_norm


def safe_normalize(tensor):
	min_val = tensor.min()
	max_val = tensor.max()
	if max_val - min_val < 1e-8:
		return torch.zeros_like(tensor)  # ou torch.ones_like(tensor), selon ta logique
	else:
		return (tensor - min_val) / (max_val - min_val)


def compute_reward_from_playersDE(game, players: list[Player], players2: list[Player], score_territory):
	N = len(players)

	if N == 0:
		return torch.zeros(5, dtype=torch.float)

	# Score brut de territoire
	scores = torch.full((N,), score_territory, dtype=torch.float)

	# Positions des joueurs
	pos1 = torch.tensor([[p.coord.x, p.coord.y] for p in players], dtype=torch.float)  # shape (N, 2)
	pos2 = torch.tensor([[p.coord.x, p.coord.y] for p in players2], dtype=torch.float)  # shape (M, 2)
	dead2 = torch.tensor([p.dead for p in players2], dtype=torch.bool)  # shape (M,)

	if dead2.all():
		reward_norm = torch.ones(N, dtype=torch.float) * 0.95
	else:
		diffs = pos1[:, None, :] - pos2[None, :, :]  # (N, M, 2)
		dists = torch.sum(torch.abs(diffs), dim=2)   # (N, M)
		dists[:, dead2] = float('inf')

		min_dists, closest_enemy_idx = torch.min(dists, dim=1)  # (N,)
		closest_enemy_pos = pos2[closest_enemy_idx]             # (N, 2)
		vec_to_enemy = closest_enemy_pos - pos1                 # (N, 2)

		inv_dists = 40.0 - (min_dists + 1e-6)

		team_color = players[0].team
		best_pos, _ = find_best_spot_numpy_general(game, team_color)
		best_pos_tensor = torch.tensor([best_pos], dtype=torch.float)  # (1, 2)

		dists_best = torch.sum(torch.abs(pos1 - best_pos_tensor), dim=1)  # (N,)
		proximity_to_closest = 40 - (dists_best + 1e-6)

		scores_norm = scores

		reward = (
			(proximity_to_closest / 40.0) * 0.6 +
			(scores_norm / 1000.0) * 0.3 +
			(inv_dists / 40.0) * 0.1
		)
		reward_norm = reward * 0.95  # (N,)

	# Étendre à 5 joueurs en dupliquant les valeurs si besoin
	reward_list = reward_norm.tolist()
	while len(reward_list) < 5:
		reward_list.append(reward_list[len(reward_list) % N])

	reward_tensor = torch.tensor(reward_list, dtype=torch.float)

	# Appliquer pénalité (reward = 0) pour les joueurs morts ou très mouillés
	for idx, p in enumerate(players):
		if p.wetness >= 100 or p.dead == 1.0:
			reward_tensor[idx] = 0.0

	return reward_tensor

def compute_reward_from_players2(game, players: list[Player], players2: list[Player], score_territory):
	N = len(players)

	if N == 0:
		return torch.zeros(5, dtype=torch.float)

	# Score brut de territoire
	scores = torch.full((N,), score_territory, dtype=torch.float)

	# Positions des joueurs
	pos1 = torch.tensor([[p.coord.x, p.coord.y] for p in players], dtype=torch.float)  # (N, 2)
	pos2 = torch.tensor([[p.coord.x, p.coord.y] for p in players2], dtype=torch.float)  # (M, 2)
	dead2 = torch.tensor([p.dead for p in players2], dtype=torch.bool)  # (M,)

	if dead2.all():
		reward_norm = torch.ones(N, dtype=torch.float) * 0.95
	else:
		# --- Nouveau : centre de masse des ennemis vivants ---
		alive_enemy_pos = pos2[~dead2]
		enemy_center = alive_enemy_pos.mean(dim=0)  # (2,)

		# Distance de chaque joueur allié au centre des ennemis
		dists_center = torch.sum(torch.abs(pos1 - enemy_center), dim=1)  # (N,)
		inv_dists_center = 40.0 - (dists_center + 1e-6)

		# Position optimale (existant)
		team_color = players[0].team
		best_pos, _ = find_best_spot_numpy_general(game, team_color)
		best_pos_tensor = torch.tensor([best_pos], dtype=torch.float)  # (1, 2)

		dists_best = torch.sum(torch.abs(pos1 - best_pos_tensor), dim=1)  # (N,)
		proximity_to_closest = 40 - (dists_best + 1e-6)

		# Normalisation du score
		scores_norm = scores 

		# Pondération de la récompense
		reward = (
			(proximity_to_closest / 40.0) * 0.1 +
			(scores_norm / 1000.0) * 0.3 +
			(inv_dists_center / 40.0) * 0.6  # Utilise le centre ennemi
		)
		reward_norm = reward * 0.95

	# Étendre à 5 joueurs en dupliquant si nécessaire
	reward_list = reward_norm.tolist()
	#while len(reward_list) < 5:
	#	reward_list.append(reward_list[len(reward_list) % N])

	# Pénalité pour joueurs morts ou mouillés
	for idx, p in enumerate(players):
		if p.wetness >= 100 or p.dead == 1.0:
			reward_list[idx] = 0.0

	reward_tensor = torch.tensor(reward_list, dtype=torch.float)

	
	return reward_tensor

def compute_reward_from_players(game, players: list[Player], players2: list[Player], score_territory):
	N = len(players)
	if N == 0:
		return torch.zeros(5, dtype=torch.float)

	grid_h, grid_w = game.height, game.width
	max_dist = grid_h + grid_w

	# Score brut de territoire (normalisé une seule fois)
	scores_norm = score_territory / 1000.0

	pos1 = torch.tensor([[p.coord.x, p.coord.y] for p in players], dtype=torch.float)
	pos2 = torch.tensor([[p.coord.x, p.coord.y] for p in players2], dtype=torch.float)
	dead2 = torch.tensor([p.dead for p in players2], dtype=torch.bool)

	if dead2.all():
		reward_norm = torch.ones(N) 
	else:
		# --- Centre des ennemis vivants ---
		alive_enemy_pos = pos2[~dead2]
		enemy_center = alive_enemy_pos.mean(dim=0)

		dists_center = torch.sum(torch.abs(pos1 - enemy_center), dim=1)
		inv_dists_center = (max_dist - dists_center) / max_dist  # entre 0 et 1

		# --- Distance au meilleur spot ---
		team_color = players[0].team
		best_pos, _ = find_best_spot_numpy_general(game, team_color)
		best_pos_tensor = torch.tensor(best_pos, dtype=torch.float)

		dists_best = torch.sum(torch.abs(pos1 - best_pos_tensor), dim=1)
		proximity_to_best = (max_dist - dists_best) / max_dist

   
		# --- Reward combiné ---
		reward = (
		   # avance vers le haut-gauche
			#0.1 * proximity_to_best +  # se rapprocher du best spot
			0.7 * inv_dists_center +   # se rapprocher des ennemis
			0.3 * scores_norm          # territoire
		)

		reward_norm = reward

	# Applique pénalités individuelles
	reward_list = reward_norm.tolist()
	for idx, p in enumerate(players):
		if p.wetness >= 100 or p.dead == 1.0:
			reward_list[idx] -= 0.2  # pénalité douce au lieu de 0 brutal

	reward_tensor = torch.tensor(reward_list, dtype=torch.float)
	reward_tensor = torch.clamp(reward_tensor, min=-1.0, max=1.0)

	return reward_tensor


def compute_reward_from_players_red(win, num_players=5):
	if win == 1:
		val = 1.0
	elif win == -1:
		val = -1.0
	else:
		val = 0.0
	return val

def compute_reward_from_players_blue(win, num_players=5):
	if win == -1:
		val = 1.0
	elif win == 1:
		val = -1.0
	else:
		val = 0.0
	return val




from torch.utils.data import Dataset

class MultiPlayerDataset(Dataset):
	def __init__(self, states, rewards, actions, values, advantages, old_prob):
		# states : liste de tenseurs ou tableaux (C, H, W)
		#self.states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])  # [N, C, H, W]
		self.states = torch.stack(states)

		# rewards, values, advantages et actions doivent �tre convertis en tenseurs float ou long 2D [N, P]
		# Si ce sont des listes de listes ou listes de tenseurs scalaires, on convertit proprement :

		# Conversion s�curis�e pour rewards
		if isinstance(rewards[0], torch.Tensor):
			self.rewards = torch.stack(rewards).float()  # [N, P] si rewards[i] est un vecteur
		else:
			self.rewards = torch.tensor(rewards, dtype=torch.float32)

		# Conversion s�curis�e pour actions (entiers)
		if isinstance(actions[0], torch.Tensor):
			self.actions = torch.stack(actions).float32()
		else:
			self.actions = torch.tensor(actions, dtype=torch.float32)

		# Conversion s�curis�e pour values
		if isinstance(values[0], torch.Tensor):
			self.values = torch.stack(values).float()
		else:
			self.values = torch.tensor(values, dtype=torch.float32)

		# Conversion s�curis�e pour advantages
		if isinstance(advantages[0], torch.Tensor):
			self.advantages = torch.stack(advantages).float()
		else:
			self.advantages = torch.tensor(advantages, dtype=torch.float32)

		if isinstance(old_prob[0], torch.Tensor):
			self.old_prob = torch.stack(old_prob).float()
		else:
			self.old_prob = torch.tensor(old_prob, dtype=torch.float32)

	def __len__(self):
		return self.states.size(0)

	def __getitem__(self, idx):
		return (
			self.states[idx],      # [C, H, W]
			self.rewards[idx],     # [P]
			self.actions[idx],     # [P]
			self.values[idx],      # [P]
			self.advantages[idx],   # [P]
			self.old_prob[idx]   # [P]
		)


def compute_advantagesA(rewards, values, gamma=0.99, tau=0.95):
	advantages = []
	gae = 0
	for t in reversed(range(len(rewards))):
		delta = rewards[t] + gamma * (values[t + 1] if t + 1 < len(values) else 0) - values[t]
		gae = delta + gamma * tau * gae
		advantages.insert(0, gae)
	return advantages

def compute_advantages(rewards, values, gamma=0.99, tau=0.95):
	rewards = torch.tensor(rewards, dtype=torch.float32)  # shape [T, N]
	values = torch.tensor(values, dtype=torch.float32)    # shape [T, N]
	
	T, N = rewards.shape
	advantages = torch.zeros_like(rewards)
	gae = torch.zeros(N)
	
	for t in reversed(range(T)):
		next_value = values[t + 1] if t + 1 < T else torch.zeros(N)
		delta = rewards[t] + gamma * next_value - values[t]
		gae = delta + gamma * tau * gae
		advantages[t] = gae
	
	return advantages.tolist()


def get_mcts_probability(current_time, max_time):
	elapsed = current_time / max_time
	return max(0.7, 0.9 - 0.2 * elapsed)  # décroît linéairement de 0.9 à 0.5


def TrainPPO():

	policy_net = PolicyNet()
	policy_net_old = PolicyNet()
	value_net = ValueNet()

	#update old
	policy_net_old.load_state_dict(policy_net.state_dict())

	# Cr�er des optimisateurs Adam pour les deux r�seaux
	policy_optimizer = torch.optim.Adam(
		policy_net.parameters(),
		lr=1e-2
	)
	
	value_optimizer = torch.optim.Adam(
		value_net.parameters(),
		lr=1e-2
	)
	
	scheduler = torch.optim.lr_scheduler.StepLR(policy_optimizer, step_size=1000, gamma=0.9)
	schedulerv = torch.optim.lr_scheduler.StepLR(value_optimizer, step_size=1000, gamma=0.9)
	
	time_tot = 0
	MAX_EPISODE = 200
	total_loss_ep = 0

	MAX_EPISODE_T = time.perf_counter() + 10 * 60
	episode = 0

	index = 0

	reward_history = []
	all_losses = []
	while episode < 50 and time.perf_counter() < MAX_EPISODE_T:
		start_time = time.perf_counter()
		
		done = [0]
		turn = 0

		rewards = []
		action = []
		values = []
		old_probs = []

		state_tab =[]
			
		grid = GridMaker.init_grid(rng)
		left_players, right_players = spawn(grid)
		game = Game(grid, left_players, right_players)

		ind = (index % 2) + 1
		index+=1				
		# Boucle principale
		I = 0
		while I < 100:
			# encoder l'�tat complet avec encode_ALL_RL (retourne un tensor)
			state_tensor = []
			

			if ind == 1:
				
				state_tensor = encode_ALL_RL(grid, game.red, game.blue, game)  # shape (canaux, H, W)
			else:
				state_tensor = encode_ALL_RL(grid, game.blue, game.red, game)

			# Pour le batch, on ajoute une dimension (batch=1)
			state_tensor_batch = state_tensor.unsqueeze(0)  # shape (1, C, H, W)

			state_tab.append(state_tensor.clone())  # sauvegarder l��tat (clone pour �viter pointer sur la m�me m�moire)

			won = -2
			#now = time.perf_counter()
			#progress = now / MAX_EPISODE_T
			#p_mcts = get_mcts_probability(now, MAX_EPISODE_T)

			#if (I % 10) < 9:
				#won = game.PlayX10Terr(ind)
				#if (I % 2) == 0:
			#	won = game.PlayX10TerrMCTS(ind)
				#else:
				#	won = game.PlayX10Terr(ind)
				
			#else:
			won, probs = game.PlayX10TerrNN_vs_MCTS(ind, policy_net_old)
			#elif (I % 4) == 2:
		#		won = game.PlayX10TerrMCTSAH(ind, policy_net_old)
			#else:
			#	won = game.PlayX_NN10AH5(policy_net_old, ind)

			reward = []
			if ind == 1:
				reward = compute_reward_from_players(game, game.red, game.blue, game.reward)
			else:
				reward = compute_reward_from_players(game, game.blue, game.red, game.reward2)
			#reward2 = compute_reward_from_players(game, complete_team(game.blue, "blue"), complete_team(game.red, "red"), game.reward2)
			combined_reward = reward #torch.cat([reward, reward2])  # shape (5,)

			action.append(game.action.copy())    # liste d�actions (5 joueurs)
			rewards.append(combined_reward.tolist())    # liste de rewards (5 joueurs)
			old_probs.append(probs)

			#game.remove_wet_players()

			# Obtenir la sortie du r�seau de valeur (valeurs par joueur)
			value_net.eval()
			with torch.no_grad():
				value_output = value_net(state_tensor_batch)  # shape: (1, 5) si batch size = 1

			# Extraire les 5 valeurs et les ajouter � la liste `values`
			values.extend(value_output.squeeze(0).tolist())  # ajoute 5 scalaires � la liste

			if won != -2:
				break
			
			I += 1
			turn += 1

		print("Simulation=", I, "/100")
		
		advantages = compute_advantages(rewards, values)
		torch.autograd.set_detect_anomaly(True)

		policy_net.train()
		policy_net_old.eval()
		value_net.train()

		total_loss = 0.0
		batch_count = 0
		for nbatch in range(32, 15, -16):
		
			# Cr�er le dataset
			dataset = MultiPlayerDataset(state_tab, rewards, action, values, advantages, old_probs)

			# Cr�er un DataLoader
			dataloader = DataLoader(dataset, batch_size=nbatch, shuffle=True)

			

			# Boucle d'entra�nement ou de test
			for batch in dataloader:
			
				state_batch, reward_batch, action_batch, value_batch, adv_batch, old_probs_batch = batch  # D�baller le batch

				if state_batch.size(0) <= 1:
					continue

				mean_reward_per_agent = reward_batch.mean(dim=0)
				#print("Mean reward per agent:", mean_reward_per_agent.cpu().numpy())
				reward_history.append(mean_reward_per_agent.cpu().numpy())

				# Les tenseurs sont d�j� en format tensor gr�ce � CustomDataset
				# Aucune conversion suppl�mentaire n'est n�cessaire ici

				# Calculer les logits et les probabilit�s
				action_logits = policy_net(state_batch) 
				action_probs = F.softmax(action_logits, dim=-1)

				##print(action_probs.shape)     # Devrait �tre [batch_size, n_actions]
				##print(action_batch.shape)     # Devrait �tre [batch_size]
				##print(action_batch)

				with torch.no_grad():
					action_logits_old = policy_net_old(state_batch)
					action_probs_old = F.softmax(action_logits_old, dim=-1)

				# R�cup�rer les probabilit�s pour les actions prises
				# `action_batch` est de taille [batch_size, n_agents]

				batch_size, n_agents = action_batch.shape

				batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, n_agents)  # [batch_size, n_agents]
				agent_indices = torch.arange(n_agents).unsqueeze(0).expand(batch_size, n_agents)    # [batch_size, n_agents]

				action_probs_taken = action_probs[batch_indices, agent_indices, action_batch]
				action_probs_old_taken = action_probs_old[batch_indices, agent_indices, action_batch]

				#print("action_prob=", action_probs)
				#print("action_prob old=", action_probs_old)

				# Calculer r_t
				#log_ratio = action_probs_taken - action_probs_old_taken
				#print(log_ratio)
				#rt = torch.exp(log_ratio)
				#rt = torch.clamp(rt, min=0, max=1.5)  # éviter valeurs trop petites ou trop grandes
				#action_probs_old_taken = torch.clamp(action_probs_old_taken, min=1e-8)
				#print("old_prob=", action_probs_old_taken)
				rt = action_probs_taken / action_probs_old_taken

				clipping_ratio = 0.1
				rt_clipped = torch.clamp(rt, 1 - clipping_ratio, 1 + clipping_ratio)

				# Normalisation des avantages
				adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
				adv_batch = adv_batch.detach()
				#adv_batch = (adv_batch - adv_batch.min()) / (adv_batch.max() - adv_batch.min() + 1e-8)

				#adv_batch = torch.clamp(adv_batch, -1, 1)

				# Calcul PPO clipped objective
				LCLIP = -torch.mean(torch.min(rt * adv_batch, rt_clipped * adv_batch))

				# Calcul de la valeur prédite
				value_output = value_net(state_batch).squeeze(-1)
				target_value = reward_batch.detach()

				# Perte valeur (MSE)
				loss_value = F.mse_loss(value_output, target_value)

				# Entropie pour exploration
				#eaction_probs = F.softmax(action_logits, dim=-1)
				entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1).mean()

				entropy_coefficient = 0.05
				value_loss_coef = 0.5

				# Perte totale combinée
				total_loss_batch = LCLIP - entropy_coefficient * entropy
				value_loss_batch = value_loss_coef * loss_value
								

				# Backpropagation 
				policy_optimizer.zero_grad()
				total_loss_batch.backward(retain_graph=True)
				torch.nn.utils.clip_grad_norm_(
					list(policy_net.parameters()) + list(value_net.parameters()), 
					0.5
				)
				policy_optimizer.step()
				scheduler.step()

				value_optimizer.zero_grad()
				value_loss_batch.backward()
				torch.nn.utils.clip_grad_norm_(
					value_net.parameters(), 
					0.5
				)
				value_optimizer.step()
				schedulerv.step()

								
								
				total_loss += total_loss_batch.item()
				batch_count += 1

				#print(f"log_ratio min/max: {log_ratio.min().item():.3f}/{log_ratio.max().item():.3f}")
				#print(f"rt min/max: {rt.min().item():.3f}/{rt.max().item():.3f}")
				#print(f"adv_batch min/max: {adv_batch.min().item():.3f}/{adv_batch.max().item():.3f}")
				#print("ADV mean/std", adv_batch.mean().item(), adv_batch.std().item())
				#print("RT min/max", rt.min().item(), rt.max().item())
				#print("Value output min/max", value_output.min().item(), value_output.max().item())
				#print("Loss LCLIP:", LCLIP.item(), "Loss value:", loss_value.item())


		

		average_loss = total_loss / batch_count if batch_count > 0 else 0.0
		
		total_loss_ep += average_loss
		avg_tot_loss_ep = total_loss_ep / (episode+1)
		all_losses.append(avg_tot_loss_ep)
		#if ((episode+1) % 10 == 0):
		print(f'episode={episode} Loss: {average_loss}/{avg_tot_loss_ep}')
		episode += 1

		#update old
		#if episode % 5 == 0:
		policy_net_old.load_state_dict(policy_net.state_dict())
		policy_net_old.eval()

		# Fin du chronom�trage
		end_time = time.perf_counter()

		# Calculer la dur�e
		execution_time = end_time - start_time
		time_tot += execution_time
		ftime_tot = str(timedelta(seconds=time_tot))
		rem_time = MAX_EPISODE_T - end_time
		rem_time_hms = str(timedelta(seconds=rem_time))


		print(f"Temps d'ex�cution : {execution_time:.2f}s/{ftime_tot}  temps restant={rem_time_hms}")

	#policy_net.display_gb()
	#save_best_nn_py(policy_net, 'best_wpod.cpp')

	torch.save({
		'model_state_dict': policy_net.state_dict()
	}, 'checkpoint6uslim.pth')

	reward_history = np.array(reward_history)

	for i in range(reward_history.shape[1]):
		plt.plot(reward_history[:, i], label=f"Agent {i}")

	plt.xlabel("Simulation step")
	plt.ylabel("Mean reward")
	plt.legend()
	plt.title("Rewards per agent")
	plt.show()

	plt.plot(all_losses)
	plt.xlabel("Episode")
	plt.ylabel("Average Loss")
	plt.title("Évolution de la perte moyenne")
	plt.grid(True)
	plt.show()
#------------------END ENCODING NN -------------------------


def Train_DPG():

	
	state_dim = 9
	action_dim = 25
	max_action = 2.0

	agent = DDPG(state_dim, action_dim, max_action)

	episodes = 0
	MAX_EPISODE_T = time.perf_counter() + 10 * 60
	time_tot = 0
	all_loss = []
	while time.perf_counter() < MAX_EPISODE_T:
		start_time = time.perf_counter()

		grid = GridMaker.init_grid(rng)
		left_players, right_players = spawn(grid)
		game = Game(grid, left_players, right_players)

		ind = (episodes % 2) + 1
		done = [0]
		episode_reward = 0
		count = 0
		agent.actor_loss = 0
		while True:
			state = 0
			if ind == 1:
				state = encode_ALL_RL(grid, game.red, game.blue, game)  # shape (canaux, H, W)
			else:
				state = encode_ALL_RL(grid, game.blue, game.red, game)

			won = game.PlayX10TerrNN_vs_MCTS(ind, agent)

			if ind == 1:
				reward = compute_reward_from_players_red(game.red, game.blue, won)
				#reward = compute_reward_from_players(game, game.red, game.blue, game.reward, won)
			else:
				reward = compute_reward_from_players_blue(game.blue, game.red, won)
				#reward = compute_reward_from_players(game, game.blue, game.red, game.reward2, won)

			action = game.actionag

			dones = 1.0 if won != -2 or count == 100 else 0.0

			next_state = 0
			if ind == 1:
				next_state = encode_ALL_RL(grid, game.red, game.blue, game)  # shape (canaux, H, W)
			else:
				next_state = encode_ALL_RL(grid, game.blue, game.red, game)


			agent.replay_buffer.append((state, action, reward, next_state, float(dones)))
			episode_reward += reward.mean().item()

			agent.train()

			count += 1
			if won != -2 or count == 100:
				break

		all_loss.append(agent.actor_loss/count)
		print(f"Episode {episodes+1}, Reward: {(episode_reward/count):.2f} / {count}")
		# Fin du chronomtrage
		end_time = time.perf_counter()

		episodes+=1

		# Calculer la dure
		execution_time = end_time - start_time
		time_tot += execution_time
		ftime_tot = str(timedelta(seconds=time_tot))
		rem_time = MAX_EPISODE_T - end_time
		rem_time_hms = str(timedelta(seconds=rem_time))

		print(f"Temps d'execution : {execution_time:.2f}s/{ftime_tot}  temps restant={rem_time_hms}")



	torch.save({
		'model_state_dict': agent.actor.state_dict()
	}, 'checkpoint6uslim.pth')

	plt.plot(agent.total_actor_loss)
	plt.xlabel("Episode")
	plt.ylabel("Average Loss")
	plt.title("Evolution de la perte moyenne")
	plt.grid(True)
	plt.show()

def Train_DQN():

	agent = DQNAgent(state_channels=11, num_actions=5)

	episodes = 0
	MAX_EPISODE_T = time.perf_counter() + 40 * 60
	time_tot = 0
	all_loss = []
	while time.perf_counter() < MAX_EPISODE_T:
		start_time = time.perf_counter()

		grid = GridMaker.init_grid(rng)
		left_players, right_players = spawn(grid)
		game = Game(grid, left_players, right_players)

		ind = 1
		done = [0]
		episode_reward = 0
		count = 0
		loss = 0
		while True:
			statess = []
			state = 0
			
			for idx, pl in enumerate(game.red):
				state = encode_ALL_RL(idx, grid, game.red, game.blue, game)  # shape (canaux, H, W)
				statess.append(state)
			

			won = game.PlayX10TerrNN_vs_Random(ind, agent)

			reward = 0
			reward = game.reward #compute_reward_from_players_red(won)
			#reward = compute_reward_from_players(game, game.red, game.blue, game.reward)
			
			action = game.actionag
			rew = []
			for a in action:
				if a == 4:
					r = reward - 100
				else:
					r = reward
				rew.append(r)

			dones = 1.0 if won != -2 or count == 100 else 0.0

			next_statess = []
			next_state = 0
			
			for idx, pl in enumerate(game.red):
				next_state = encode_ALL_RL(idx, grid, game.red, game.blue, game)  # shape (canaux, H, W)
				next_statess.append(next_state)
		
							
			r = 0
			
			for idx, pl in enumerate(game.red):
					
				#if won == 1:rew = 1
				if pl.wetness >= 100:rew[idx] = -1
				#elif won == -1:rew = -1
				#else:rew=  reward#[idx].item()
				agent.replay.push(statess[idx], action[idx], rew[idx], next_statess[idx], float(dones))
				r+=rew[idx]
			if len(game.red) > 0:r /= len(game.red)
			
			game.remove_wet_players()

			episode_reward += r
						
			count += 1
			if won != -2 or count == 100:
				break

		for i in range(30):
			l = agent.optimize()
			if l is not None:
				loss += l

		all_loss.append(loss/30)
		print(f"Episode {episodes+1}, Reward: {(episode_reward/count):.2f} / {count}, eps={agent.eps}")
		# Fin du chronomtrage
		end_time = time.perf_counter()

		episodes+=1

		# Calculer la dure
		execution_time = end_time - start_time
		time_tot += execution_time
		ftime_tot = str(timedelta(seconds=time_tot))
		rem_time = MAX_EPISODE_T - end_time
		rem_time_hms = str(timedelta(seconds=rem_time))

		print(f"Temps d'execution : {execution_time:.2f}s/{ftime_tot}  temps restant={rem_time_hms}")



	torch.save({
		'model_state_dict': agent.q_net.state_dict()
	}, 'checkpoint6uslim_dqnonered.pth')

	plt.plot(all_loss)
	plt.xlabel("Episode")
	plt.ylabel("Average Loss")
	plt.title("Evolution de la perte moyenne")
	plt.grid(True)
	plt.show()


def draw_grid(grid, left_players, right_players):
	w, h = grid.width, grid.height
	for y in range(h):
		for x in range(w):
			cell = grid.get(x, y)
			rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
			t = cell.get_type()
			if t == Tile.TYPE_FLOOR:
				color = WHITE
			elif t == Tile.TYPE_LOW_COVER:
				color = GREY
			elif t == Tile.TYPE_HIGH_COVER:
				color = DARK_GREY
			else:
				color = BLACK
			pygame.draw.rect(screen, color, rect)
			pygame.draw.rect(screen, BLACK, rect, 1)

	for player in left_players:
		if player.wetness >= 100: continue
		center = (player.coord.get_x() * CELL_SIZE + CELL_SIZE // 2,
				  player.coord.get_y() * CELL_SIZE + CELL_SIZE // 2)
		pygame.draw.circle(screen, (255, 0, 0), center, CELL_SIZE // 3)

	for player in right_players:
		if player.wetness >= 100: continue
		center = (player.coord.get_x() * CELL_SIZE + CELL_SIZE // 2,
				  player.coord.get_y() * CELL_SIZE + CELL_SIZE // 2)
		pygame.draw.circle(screen, (0, 0, 255), center, CELL_SIZE // 3)

	for player in left_players + right_players:
		if player.wetness >= 100: continue
		if player.thx != -1 and player.thy != -1:
			target_center = (player.thx * CELL_SIZE + CELL_SIZE // 2,
							 player.thy * CELL_SIZE + CELL_SIZE // 2)
			pygame.draw.circle(screen, (255, 255, 0), target_center, CELL_SIZE // 6)


TRAIN_PPO = False
PLAY_NN =  True
PLAY_MCTS = False

if TRAIN_PPO:
	rng = random.Random()
	Train_DQN()

else:
	pygame.init()
	screen = pygame.display.set_mode((30 * CELL_SIZE, 20 * CELL_SIZE))  # Choisir une taille suffisante
	pygame.display.set_caption("Grille avec obstacles")
	clock = pygame.time.Clock()
	rng = random.Random()

	winr = 0
	winb = 0
	winnn = 0

	index = 0
	# Boucle principale de parties
	running = True
	while running:
		#  Reinitialiser la partie
		grid = GridMaker.init_grid(rng)
		left_players, right_players = spawn(grid)
		game = Game(grid, left_players, right_players)
		if PLAY_NN or PLAY_MCTS:
			game.init_NNUSNW()

		ind = 1
		if ind == 1:
			print("PLAY RED", ind)
		else:
			print("PLAY BLUE", ind)

		index+=1

		# Boucle d'une partie
		won = -2
		ind_sim = 0
		while won == -2 and running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False

			screen.fill(WHITE)
			draw_grid(grid, game.red, game.blue)

			if not PLAY_NN and not PLAY_MCTS:
				won = game.Play()
			elif PLAY_MCTS:
				won = game.PlayX10TerrMCTS(ind) #PlayX_NN10NMCTS(ind)PlayX10TerrMCTS
			else:
				won = game.PlayX_NN10N(ind)
							

			ind_sim += 1
			if ind_sim == 100:
				sc = game.rscore - game.bscore
				if sc > 0:
					won = 1
				elif sc < 0:
					won = -1
				else:
					won = -3

			if won == 1:
				winr+=1
				if ind == 1:
					winnn+=1
			elif won == -1:
				winb += 1
				if ind == 2:
					winnn+=1

			if won != -2:
				red_pct = 100 * winr / index
				blue_pct = 100 * winb / index
				print(f"RED = {winr} ({red_pct:.2f}%)   BLUE = {winb} ({blue_pct:.2f}%) / {index} on {ind_sim} simulations")
						
				nn_pct = 100 * winnn / index
				print(f"NN = {winnn} ({nn_pct:.2f}%")



			

			pygame.display.flip()
			clock.tick(60)
			#time.sleep(0.1)

		# Annoncer le gagnant
		if won == 1:
			print(" Red wins!")
		elif won == -1:
			print(" Blue wins!")
		elif won != -2:
			print(f" Unexpected winner: {won}")

		# Petite pause avant la prochaine partie
		#time.sleep(2)

pygame.quit()

