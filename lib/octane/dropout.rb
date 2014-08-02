module Octane
	module Dropout
		def enable_all_units
			(0...@layers.size).each {|layer|
				enable_unit(layer, (0...@layers[layer].size).to_a)
			}
		end

		def disable_unit(layer, units)
			if units.is_a?(Array)
				units.map{|unit| disable_unit(layer, unit)}
			else
				unit=units
				unless @layers[layer][unit].disabled
					next_layer=layer+1
					prev_layer=layer-1

					@layers[layer][unit].disabled=true

					@layers[next_layer].each_with_index {|neuron, index|
						neuron.ensure_disabled(unit)
					}
				end
			end

		end

		def enable_unit(layer,units)
			if units.is_a?(Array)
				units.map{|unit| enable_unit(layer, unit)}
			else
				unit=units
				if @layers[layer][unit].disabled
					next_layer=layer+1
					prev_layer=layer-1

					@layers[layer][unit].disabled=false

					@layers[next_layer].each_with_index {|neuron, index|
						neuron.ensure_enabled(unit)
					}
				end
			end
		end

	end
end
