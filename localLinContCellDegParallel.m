function output = localLinContCellDegParallel(pidx, tidx, freqmode)
    % read the HDF5 file
    tic
    filename = sprintf('models/Linear/singlecell_%d_%d.h5', pidx, tidx);
    modelslopes = h5read(filename, '/modelslopes');
    modelintercept = h5read(filename, '/modelintercept');
    ranges = h5read(filename, '/ranges');
    validp = h5read(filename, '/validp');
    validtheta = h5read(filename, '/validtheta');
    tcomp1 = toc;
    disp(['***********computation time of reading the HDF5 file: ',num2str(tcomp1)]);


    % Parameters --------------------------------------------------------------

    max_linear_model = modelintercept;
    min_linear_model = modelintercept;
    if modelslopes(1) > 0
        max_linear_model = max_linear_model + modelslopes(1) * validp(2);
        min_linear_model = min_linear_model + modelslopes(1) * validp(1);
    else
        max_linear_model = max_linear_model + modelslopes(1) * validp(1);
        min_linear_model = min_linear_model + modelslopes(1) * validp(2);
    end

    if modelslopes(2) > 0
        max_linear_model = max_linear_model + modelslopes(2) * validtheta(2);
        min_linear_model = min_linear_model + modelslopes(2) * validtheta(1);
    else
        max_linear_model = max_linear_model + modelslopes(2) * validtheta(1);
        min_linear_model = min_linear_model + modelslopes(2) * validtheta(2);
    end

    params.tFinal = 1.0; %final time
    p_cell_lb = -10 + (double(pidx))*(10 - (-10)) / 128;
    p_cell_ub = -10 + (double(pidx)+1)*(10 - (-10)) / 128;
    theta_cell_lb = -30 + (double(tidx))*(30 - (-30)) / 128;
    theta_cell_ub = -30 + (double(tidx)+1)*(30 - (-30)) / 128;
    params.R0 = zonotope(interval([p_cell_lb;theta_cell_lb],[p_cell_ub;theta_cell_ub])+[0.0;0.0]); % heading error is in degree
    if strcmp(freqmode, 'inf')
        params.U = zonotope(interval(ranges(1), ranges(2)));
    elseif strcmp(freqmode, 'fixed')
        params.U = zonotope(interval(double(min_linear_model+ ranges(1)), double(max_linear_model + ranges(2))));
    else
        disp('Invalid frequency mode')
        return
    end
    

    % Reachability Settings ---------------------------------------------------

    options.timeStep = 0.01;
    options.taylorTerms = 4;
    options.zonotopeOrder = 50;
    options.alg = 'lin';
    options.tensorOrder = 2;

    options.lagrangeRem.simplify = 'simplify';

    % System Dynamics ---------------------------------------------------------

    v = 5;
    L = 5;

    [m1, m2] = deal(modelslopes(1), modelslopes(2));
    bias = modelintercept;

    % uncertainty on the control output
    if strcmp(freqmode, 'inf')
        fun = @(x,u) [v*sin(pi/180*x(2)); ...
                    180/pi*v/L*tan(pi/180*(m1*x(1)+m2*x(2)+bias+u(1)))];
    elseif strcmp(freqmode, 'fixed')
        fun = @(x,u) [v*sin(pi/180*x(2)); ...
                    180/pi*v/L*tan(pi/180*(u(1)))];
    else
        disp('Invalid frequency mode')
        return
    end

    sys = nonlinearSys(sprintf('sy_s_%d_%d', pidx, tidx), fun); % initialize the system


    % Reachability Analysis ---------------------------------------------------

    tic
    R = reach(sys, params, options);
    tComp = toc;
    disp(['***********computation time of reachable set: ',num2str(tComp)]);

    tic
    output = true;

    validreg = interval([double(validp(1));double(validtheta(1))],[double(validp(2));double(validtheta(2))]);
    finalR = query(R, 'finalSet');
    if strcmp(freqmode, 'inf')
        containedRes = contains(validreg, finalR);
        if ~(containedRes)
            disp("Not contained")
            output = false;
        end

        if output
            for i=1:length(R(1).timeInterval.set)
                containedRes = contains(validreg, R(1).timeInterval.set{i,1});
                if ~(containedRes)
                    disp("Not contained")
                    output = false;
                    break
                end
            end
        end
    end
    tcomp2 = toc;
    disp(['***********computation time of containment checking: ',num2str(tcomp2)]);

    maxs = max(vertices(finalR), [], 2);
    mins = min(vertices(finalR), [], 2);
    p_reach_lb = mins(1);
    p_reach_ub = maxs(1);
    theta_reach_lb = mins(2);
    theta_reach_ub = maxs(2);
    
    if p_reach_lb < -10 || p_reach_ub > 10 || theta_reach_lb < -30 || theta_reach_ub > 30
        output = true;  % it does not need to bloat further
    end



    % next state array ----------------------------------------------------------------
    tic
    if output

        p_bins = linspace(-10, 10, 128 + 1);
        p_lbs = single(p_bins(1:end-1));
        p_ubs = single(p_bins(2:end));

        theta_bins = linspace(-30, 30, 128 + 1);
        theta_lbs = single(theta_bins(1:end-1));
        theta_ubs = single(theta_bins(2:end));

        % For p_reach_lb
        if p_reach_lb < p_lbs(1)
            p_index_lb = -1;
        else
            p_index_lb = find(p_lbs <= p_reach_lb, 1, 'last');
        end
        % For p_reach_ub
        if p_reach_ub > p_ubs(end)
            p_index_ub = -1;
        else
            p_index_ub = find(p_ubs >= p_reach_ub, 1, 'first');
        end
        % For theta_reach_lb
        if theta_reach_lb < theta_lbs(1)
            theta_index_lb = -1;
        else
            theta_index_lb = find(theta_lbs <= theta_reach_lb, 1, 'last');
        end
        % For theta_reach_ub
        if theta_reach_ub > theta_ubs(end)
            theta_index_ub = -1;
        else
            theta_index_ub = find(theta_ubs >= theta_reach_ub, 1, 'first');
        end

        reachable_range = [p_reach_lb, p_reach_ub; theta_reach_lb, theta_reach_ub];  
        reachable_range_idx = [p_index_lb, p_index_ub; theta_index_lb, theta_index_ub];

        % Check if models folder exists or not
        if exist('models/reachgraph', 'dir') == 7
            disp('models/reachgraph folder already exists')
        else
            % Create models folder
            mkdir('models/reachgraph')
        end

        filename = sprintf('models/reachgraph/reachGraph_%d_%d.mat',pidx, tidx);
        save(filename, 'reachable_range', 'reachable_range_idx');

    end
    tcomp3 = toc;
    disp(['***********computation time of saving the data: ',num2str(tcomp3)]);

    % Visualization -----------------------------------------------------------

    tic
    vis_flag = false;
    if vis_flag
        % Simulation --------------------------------------------------------------
        simOpt.points = 10;
        simRes = simulateRandom(sys, params, simOpt);
        % Simulation --------------------------------------------------------------

        figure; hold on; box on;
        grid on;

        plot(validreg)

        % plot reachable sets
        plot(finalR)
        % plot(R);

        % plot initial set
        plot(params.R0,[1,2],'k','FaceColor','w');

        % plot simulation results     
        plot(simRes,[1,2],'Marker','.');

        xlabel('cross track error(Meter)')
        ylabel('heading error(Degree)')
        figfilename = sprintf('Fig/reach_oneCell_%d_%d.png', pidx, tidx);
        saveas(gcf, figfilename);
    end
    tcomp4 = toc;
    disp(['***********computation time of visualization: ',num2str(tcomp4)]);

end