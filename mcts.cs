public void PlayS(Game game)
{
    Stopwatch stopwatch = Stopwatch.StartNew();

    int[,] directions = new int[,] { { 0, -1 }, { -1, 0 }, { 1, 0 }, { 0, 1 } };

    int max_depth = 0;


    List<Hit> opp = new List<Hit>();
    List<State> root = new List<State>();
    foreach (var a in game.my_agent)
    {
        Hit hp = new Hit(a.Value.x, a.Value.y);
        root.Add(new State(null, hp));
        root.Last().id = a.Value.agent_id;

        Hit h = new Hit(-1, -1);
        int mind = 20000000;
        int maxs = -2000000;

        foreach (var o in game.opp_agent)
        {
            int d = game.distance[hp.y * game.width + hp.x][o.Value.y * game.width + o.Value.x];
            int sc = (6 - o.Value.splash_bombs) * 100 - d;
            if (sc > maxs)
            {
                maxs = sc;
                mind = d;
                h.x = o.Value.x;
                h.y = o.Value.y;
            }
        }

        root.Last().hit2 = h;


    }

    List<State> rooto = new List<State>();
    foreach (var a in game.opp_agent)
    {
        Hit hp = new Hit(a.Value.x, a.Value.y);
        rooto.Add(new State(null, hp));
        rooto.Last().id = a.Value.agent_id;

        Hit h = new Hit(-1, -1);
        int mind = 20000000;
        int maxs = -2000000;

        foreach (var o in game.my_agent)
        {
            int d = game.distance[hp.y * game.width + hp.x][o.Value.y * game.width + o.Value.x];
            int sc = (6 - o.Value.splash_bombs) * 100 - d;
            if (sc > maxs)
            {
                maxs = sc;
                mind = d;
                h.x = o.Value.x;
                h.y = o.Value.y;
            }
        }

        rooto.Last().hit2 = h;


    }



    int turn = 0;

    NODE = 0;

    int MAX_DEPTH = 2;//(int)Interpolate((double)game.my_agent.Count + game.opp_agent.Count);

    //C = (double)((game.my_agent.Count + game.opp_agent.Count) - 3) / 10.0;
    //if(C <= 0.0)C = 0.1;


    while ((int)stopwatch.ElapsedMilliseconds <= timeLimit)
    {
        List<Game> lgamepl = new List<Game>();
        List<State> lnode = new List<State>();
        for (int i = 0; i < root.Count; ++i)
        {

            Game gamepl = game.Clone();
            gamepl.x1 = root[i].hit.x;
            gamepl.y1 = root[i].hit.y;
            gamepl.x2 = root[i].hit2.x;
            gamepl.y2 = root[i].hit2.y;
            lgamepl.Add(gamepl);
            lnode.Add(root[i]);

        }

        List<Game> lgameplo = new List<Game>();
        List<State> lnodeo = new List<State>();
        for (int i = 0; i < rooto.Count; ++i)
        {

            Game gamepl = game.Clone();
            gamepl.x1 = rooto[i].hit.x;
            gamepl.y1 = rooto[i].hit.y;
            gamepl.x2 = rooto[i].hit2.x;
            gamepl.y2 = rooto[i].hit2.y;
            lgameplo.Add(gamepl);
            lnodeo.Add(rooto[i]);

        }

        Game gamea = game.Clone();


        for (int depth = 0; depth < MAX_DEPTH; ++depth)
        {
            if ((int)stopwatch.ElapsedMilliseconds > timeLimit) break;
            for (int i = 0; i < root.Count; ++i)
            {
                if ((int)stopwatch.ElapsedMilliseconds > timeLimit) break;
                var ag = gamea.my_agent[root[i].id];
                if (ag.wetness >= 100) continue;

                if (lnode[i].child.Count == 0)
                {
                    List<Hit> hits = lgamepl[i].GetHits(lnode[i].hit.x, lnode[i].hit.y, lgamepl[i].grid, gamea);
                    Expand(lnode[i], hits);

                }

                if (lnode[i].child.Count == 0) continue;
                lnode[i] = Selection(lnode[i]);
                lgamepl[i].PlayP1(lnode[i].hit);
                Agent agent = gamea.my_agent[root[i].id].Clone();
                agent.x = lnode[i].hit.x;
                agent.y = lnode[i].hit.y;
                gamea.my_agent[root[i].id] = agent;


            }

            for (int i = 0; i < rooto.Count; ++i)
            {
                if ((int)stopwatch.ElapsedMilliseconds > timeLimit) break;
                var ag = gamea.opp_agent[rooto[i].id];
                if (ag.wetness >= 100) continue;

                if (lnodeo[i].child.Count == 0)
                {
                    List<Hit> hits = lgameplo[i].GetHitsO(lnodeo[i].hit.x, lnodeo[i].hit.y, lgameplo[i].grid, gamea);
                    Expand(lnodeo[i], hits);

                }

                if (lnodeo[i].child.Count == 0) continue;
                lnodeo[i] = Selection(lnodeo[i]);
                lgameplo[i].PlayP1(lnodeo[i].hit);
                Agent agent = gamea.opp_agent[rooto[i].id].Clone();
                agent.x = lnodeo[i].hit.x;
                agent.y = lnodeo[i].hit.y;
                gamea.opp_agent[rooto[i].id] = agent;


            }

        }

        if ((int)stopwatch.ElapsedMilliseconds > timeLimit) break;

        int my_count = 0, opp_count = 0;
        for (int y = 0; y < game.height; ++y)
        {
            for (int x = 0; x < game.width; ++x)
            {
                int best_dist1 = int.MaxValue;
                int best_dist2 = int.MaxValue;

                for (int a = 0; a < root.Count; ++a)
                {
                    var ha = lnode[a].hit;
                    int da = Math.Abs(y - ha.y) + Math.Abs(x - ha.x);
                    var ag = gamea.my_agent[root[a].id];
                    if (ag.wetness >= 100) continue;
                    if (ag.wetness >= 50) da *= 2;
                    best_dist1 = Math.Min(best_dist1, da);
                }

                /*foreach (var kvp in gamea.opp_agent) {
                    var a = kvp.Value;
                    int da = Math.Abs(y - a.y) + Math.Abs(x - a.x);
                    if (a.wetness >= 50) da *= 2;
                    best_dist2 = Math.Min(best_dist2, da);
                }*/

                for (int a = 0; a < rooto.Count; ++a)
                {
                    var ha = lnodeo[a].hit;
                    int da = Math.Abs(y - ha.y) + Math.Abs(x - ha.x);
                    var ag = gamea.opp_agent[rooto[a].id];
                    if (ag.wetness >= 100) continue;
                    if (ag.wetness >= 50) da *= 2;
                    best_dist2 = Math.Min(best_dist2, da);
                }

                if (best_dist1 < best_dist2) my_count++;
                else if (best_dist2 < best_dist1) opp_count++;
            }
        }

        int r = my_count - opp_count;
        if (r > 0) gamea.mscore += r;
        else gamea.oscore += -r;

        // Si r > 0, on calcule un score proportionnel (max 1.0), sinon on met 0
        double score2 = (r > 0) ? (double)r / 100.0 : 0.0;

        // Clamp entre 0.0 et 1.0
        score2 = Math.Min(1.0, Math.Max(0.0, score2));

        double alpha = 0.0, beta = 0.0, omega = 0.0, theta = 0.0, phi = 0.5;

        double score_total = 0.0;
        List<double> lmyscore = new List<double>();
        List<double> loppscore = new List<double>();

        for (int i = 0; i < root.Count; ++i)
        {
            if (gamea.my_agent[root[i].id].wetness >= 100)
            {
                Backpropagation(lnode[i], -1.0);
                continue;
            }

            if ((int)stopwatch.ElapsedMilliseconds > timeLimit) break;
            double score = 0, score3 = 0, score4 = 0;

            /*if(gamea.my_agent[root[i].id].splash_bombs > 0 || gamea.my_agent[root[i].id].wetness < 50){
                alpha = 0.7;//territoire
                beta = 0.15;//distance ennemi
                omega = 0.0;//couverture
                theta = 0.15;//spacing
            }
            else{
                alpha = 0.2;
                beta = 0.33;
                omega = 0.33;
                theta = 0.13;
            }*/

            if ((gamea.mscore - gamea.oscore) > 100)
            {
                // très défensif
                alpha = 0.1;
                beta = 0.0;
                omega = 0.9;
                theta = 0.0;
                phi = 0.55;
            }
            else if (gamea.my_agent[root[i].id].splash_bombs > 0 || (gamea.mscore - gamea.oscore) < 100)
            {
                // offensif
                alpha = 0.6;
                beta = 0.25;
                omega = 0.0;
                theta = 0.15;
            }
            else if (gamea.my_agent[root[i].id].wetness < 40)
            {
                // neutre
                alpha = 0.5;
                beta = 0.2;
                omega = 0.0;
                theta = 0.3;
            }
            else
            {
                // très défensif
                alpha = 0.2;
                beta = 0.25;
                omega = 0.4;
                theta = 0.15;
            }


            /*double health = 1.0 - gamea.my_agent[root[i].id].wetness / 100.0;
            double hasBomb = gamea.my_agent[root[i].id].splash_bombs > 0 ? 1.0 : 0.0;
            double threat = 1.0 - health * 0.7 + (1.0 - hasBomb) * 0.3;

            alpha = Lerp(0.2, 0.7, 1.0 - threat);
            beta  = Lerp(0.33, 0.15, 1.0 - threat);
            omega = Lerp(0.33, 0.0, 1.0 - threat);
            theta = 1.0 - (alpha + beta + omega);  // reste*/


            int d = game.distance[lgamepl[i].y1 * lgamepl[i].width + lgamepl[i].x1][lgamepl[i].y2 * lgamepl[i].width + lgamepl[i].x2];
            if (d == int.MaxValue)
            {
                score = 0;
            }
            else
            {
                score = 100 - d;
            }

            score /= 100.0;
            score = Math.Clamp(score, 0.0, 1.0);

            double cover = -2000000;
            int counta = 0;
            for (int j = 0; j < 4; ++j)
            {
                Agent ag = gamea.my_agent[root[i].id];
                int edx = ag.x + directions[j, 0];
                int edy = ag.y + directions[j, 1];

                if (edx < 0 || edx >= game.width || edy < 0 || edy >= game.height) continue;
                if (game.grid[edy][edx].tile_type > 0)
                {
                    foreach (var kvp in game.opp_agent)
                    {
                        var a = kvp.Value;
                        if ((ag.x < edx && a.x > edx) ||
                            (ag.x > edx && a.x < edx) ||
                            (ag.y < edy && a.y > edy) ||
                            (ag.y > edy && a.y < edy))
                        {
                            ++counta;

                        }

                    }
                }

                cover += game.grid[edy][edx].tile_type * counta;
            }

                
            score3 = cover / 20.0;
            score3 = Math.Min(1.0, Math.Max(0.0, score3));

            double spacingPenalty = 0;

            var myAgentsList = gamea.my_agent.Values.ToList();
            for (int ii = 0; ii < myAgentsList.Count; ii++)
            {
                for (int j = ii + 1; j < myAgentsList.Count; j++)
                {
                    var a1 = myAgentsList[ii];
                    var a2 = myAgentsList[j];
                    int dist = Math.Abs(a1.x - a2.x) + Math.Abs(a1.y - a2.y);

                    if (dist < 2)  // Distance 0 ou 1
                    {
                        spacingPenalty += 0.1 * (2 - dist); // Plus proche → plus de pénalité
                    }
                }
            }

            score4 = 1.0 - spacingPenalty;
            score4 = Math.Min(1.0, Math.Max(0.0, score4));


            double score5 = 0.0;
            double damage = 0.0;

            //if (gamea.my_agent[root[i].id].wetness > 40 && gamea.my_agent[root[i].id].splash_bombs == 0)
            //{
            foreach (var oa in gamea.opp_agent)
            {
                Agent ag = gamea.my_agent[root[i].id];
                int di = game.distance[oa.Value.y * game.width + oa.Value.x][ag.y * game.width + ag.x];

                int wetness = oa.Value.wetness;
                int dsh = game.sopp_agent[oa.Value.agent_id].optimal_range;

                if (di <= dsh && oa.Value.cooldown == 0)
                {
                    damage += game.sopp_agent[oa.Value.agent_id].soaking_power;

                }
            }


            if (gamea.my_agent[root[i].id].wetness > 0)
                score5 = damage / (101.0 - (double)gamea.my_agent[root[i].id].wetness);
            else
                score5 = 1.0;

            Agent agent = gamea.my_agent[root[i].id].Clone();
            agent.wetness += (int)damage;
            if (agent.wetness > 100) agent.wetness = 100;
            gamea.my_agent[root[i].id] = agent;

            score5 = Math.Min(1.0, Math.Max(0.0, score5));

            //}

            double scoref = (score2 * alpha + score * beta + score3 * omega + score4 * theta) - score5 * phi; //(score2 > 0) ? score2 : score;
            score_total += scoref;
            lmyscore.Add(scoref);

            Backpropagation(lnode[i], scoref);
        }

        for (int i = 0; i < rooto.Count; ++i)
        {
            if (gamea.opp_agent[rooto[i].id].wetness >= 100)
            {
                Backpropagation(lnodeo[i], -1.0);
                continue;
            }

            if ((int)stopwatch.ElapsedMilliseconds > timeLimit) break;
            double score = 0, score3 = 0, score4 = 0;



            if ((gamea.oscore - gamea.mscore) > 100)
            {
                // très défensif
                alpha = 0.1;
                beta = 0.0;
                omega = 0.9;
                theta = 0.0;
                phi = 0.55;
            }
            else if (gamea.opp_agent[rooto[i].id].splash_bombs > 0 || (gamea.oscore - gamea.mscore) < 100)
            {
                // offensif
                alpha = 0.6;
                beta = 0.25;
                omega = 0.0;
                theta = 0.15;
            }
            else if (gamea.opp_agent[rooto[i].id].wetness < 40)
            {
                // neutre
                alpha = 0.5;
                beta = 0.2;
                omega = 0.0;
                theta = 0.3;
            }
            else
            {
                // très défensif
                alpha = 0.2;
                beta = 0.25;
                omega = 0.4;
                theta = 0.15;
            }




            int d = game.distance[lgameplo[i].y1 * lgameplo[i].width + lgameplo[i].x1][lgameplo[i].y2 * lgameplo[i].width + lgameplo[i].x2];
            if (d == int.MaxValue)
            {
                score = 0;
            }
            else
            {
                score = 100 - d;
            }

            score /= 100.0;
            score = Math.Clamp(score, 0.0, 1.0);

            double cover = -2000000;
            int counta = 0;
            for (int j = 0; j < 4; ++j)
            {
                Agent ag = gamea.opp_agent[rooto[i].id];
                int edx = ag.x + directions[j, 0];
                int edy = ag.y + directions[j, 1];

                if (edx < 0 || edx >= game.width || edy < 0 || edy >= game.height) continue;
                if (game.grid[edy][edx].tile_type > 0)
                {
                    foreach (var kvp in game.my_agent)
                    {
                        var a = kvp.Value;
                        if ((ag.x < edx && a.x > edx) ||
                            (ag.x > edx && a.x < edx) ||
                            (ag.y < edy && a.y > edy) ||
                            (ag.y > edy && a.y < edy))
                        {
                            ++counta;

                        }

                    }
                }

                cover += game.grid[edy][edx].tile_type * counta;
            }




            //score3 = cover / (double)(game.opp_agent.Count * 100);
            score3 = cover / 20.0;
            score3 = Math.Min(1.0, Math.Max(0.0, score3));

            double spacingPenalty = 0;

            var myAgentsList = gamea.opp_agent.Values.ToList();
            for (int ii = 0; ii < myAgentsList.Count; ii++)
            {
                for (int j = ii + 1; j < myAgentsList.Count; j++)
                {
                    var a1 = myAgentsList[ii];
                    var a2 = myAgentsList[j];
                    int dist = Math.Abs(a1.x - a2.x) + Math.Abs(a1.y - a2.y);

                    if (dist < 2)  // Distance 0 ou 1
                    {
                        spacingPenalty += 0.1 * (2 - dist); // Plus proche → plus de pénalité
                    }
                }
            }

            score4 = 1.0 - spacingPenalty;
            score4 = Math.Min(1.0, Math.Max(0.0, score4));


            double score5 = 0.0;
            double damage = 0.0;

           
            foreach (var oa in gamea.my_agent)
            {
                Agent ag = gamea.opp_agent[rooto[i].id];
                int di = game.distance[oa.Value.y * game.width + oa.Value.x][ag.y * game.width + ag.x];

                int wetness = oa.Value.wetness;
                int dsh = game.smy_agent[oa.Value.agent_id].optimal_range;

                if (di <= dsh && oa.Value.cooldown == 0)
                {
                    damage += game.smy_agent[oa.Value.agent_id].soaking_power;

                }
            }


            if (gamea.opp_agent[rooto[i].id].wetness > 0)
                score5 = damage / (101.0 - (double)gamea.opp_agent[rooto[i].id].wetness);
            else
                score5 = 1.0;

            Agent agent = gamea.opp_agent[rooto[i].id].Clone();
            agent.wetness += (int)damage;
            if (agent.wetness > 100) agent.wetness = 100;
            gamea.opp_agent[rooto[i].id] = agent;

            score5 = Math.Min(1.0, Math.Max(0.0, score5));


            double scoref = (score2 * alpha + score * beta + score3 * omega + score4 * theta) - score5 * phi; //(score2 > 0) ? score2 : score;
            score_total += scoref;
            loppscore.Add(scoref);

            Backpropagation(lnodeo[i], scoref);
        }

     
        ++turn;




    }

    List<Hit> player = new List<Hit>();
    for (int r = 0; r < root.Count; ++r)
    {
        double maxis = double.NegativeInfinity;
        int indexc = -1;
        State noder = root[r];
        for (int i = 0; i < noder.child.Count; ++i)
        {
            if (noder.child[i].n == 0)
            {
                Console.Error.WriteLine("i=0 " + noder.child[i].hit.x + " " + noder.child[i].hit.y);
                continue;
            }
            double score = noder.child[i].score / noder.child[i].n;
            Console.Error.WriteLine("i=" + i + " " + score);
            if (score > maxis)
            {
                maxis = score;
                indexc = i;
            }
        }

        if (indexc != -1)
            player.Add(noder.child[indexc].hit);
        else
            player.Add(root[r].hit);


    }

    
}

double Interpolate(double x)
{
    double x0 = 2, y0 = 4;
    double x1 = 10, y1 = 2;

    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}


public static double Lerp(double a, double b, double t)
{
    return a + (b - a) * t;
}

public List<Hit> GetValidNeighbors(List<List<Tile>> grid)
{
    int height = grid.Count;
    int width = grid[0].Count;
    List<Hit> hits = new List<Hit>();

    int[] dx = new int[] { 0, 1, 0, -1 };
    int[] dy = new int[] { -1, 0, 1, 0 };

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (grid[y][x].tile_type > 0)
            {
                for (int d = 0; d < 4; d++)
                {
                    int nx = x + dx[d];
                    int ny = y + dy[d];

                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                    if (grid[y][x].tile_type > 0) continue;

                    hits.Add(new Hit(nx, ny));

                }
            }
        }
    }

    return hits;
}
