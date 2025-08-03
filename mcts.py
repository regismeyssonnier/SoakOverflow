# -*- coding: latin-1 -*-
import math
import random
import time

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
		self.time_limit = 35  # ms (� g�rer selon ton environnement)
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
			if indb == 'red' and r > 0:score = r / 100
			if indb == 'blue' and r < 0:score = -r / 100

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

				score3 = max(0.0, min(1.0, cover / 20.0))

				# P�nalit� d'espacement : �vite que les agents soient trop proches
				spacing_penalty = 0.0
				my_agents_list = my_agent
				for ii in range(len(my_agents_list)):
					for jj in range(ii + 1, len(my_agents_list)):
						a1, a2 = my_agents_list[ii], my_agents_list[jj]
						dist = abs(a1.coord.x - a2.coord.x) + abs(a1.coord.y - a2.coord.y)
						if dist < 2:
							spacing_penalty += 0.1 * (2 - dist)

				score4 = max(0.0, min(1.0, 1.0 - spacing_penalty))

				# Estimation des d�g�ts potentiels
				damage = 0.0
				for oa in opp_agent:
					dist_to_agent = sim_game.graph.distance[oa.coord.y * sim_game.grid.width + oa.coord.x][agent.coord.y * sim_game.grid.width + agent.coord.x]
					if dist_to_agent <= oa.optimalRange and oa.cooldown == 0:
						damage += oa.soakingPower

				if agent.wetness > 0:
					score5 = damage / (101.0 - float(agent.wetness))
				else:
					score5 = 1.0

				
				score5 = max(0.0, min(1.0, score5))

				scoref = (score2 * alpha + score * beta + score3 * omega + score4 * theta) - score5 * phi
				
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
